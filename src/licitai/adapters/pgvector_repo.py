from datetime import datetime
from decimal import Decimal
from typing import Any, cast

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CursorResult,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from licitai.domain.models import DocumentoChunk, EditalMetadata, ResultadoBuscaHibrida


class Base(DeclarativeBase):
    pass


class EditalORM(Base):
    __tablename__ = "editais"

    numero_controle_pncp: Mapped[str] = mapped_column(String(80), primary_key=True)
    orgao_cnpj: Mapped[str] = mapped_column(String(14), nullable=False, index=True)
    orgao_nome: Mapped[str] = mapped_column(String(255), nullable=False)
    uf: Mapped[str] = mapped_column(String(2), nullable=False, index=True)
    municipio: Mapped[str | None] = mapped_column(String(120), nullable=True)
    modalidade_nome: Mapped[str] = mapped_column(String(100), nullable=False)
    objeto_compra: Mapped[str] = mapped_column(Text, nullable=False)
    valor_total_estimado: Mapped[Decimal | None] = mapped_column(Numeric(18, 2), nullable=True)
    data_publicacao_pncp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    link_sistema_origem: Mapped[str | None] = mapped_column(Text, nullable=True)


class DocumentoChunkORM(Base):
    __tablename__ = "documento_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    numero_controle_pncp: Mapped[str] = mapped_column(
        String(80),
        ForeignKey("editais.numero_controle_pncp", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    arquivo_nome: Mapped[str] = mapped_column(String(255), nullable=False)
    numero_pagina: Mapped[int] = mapped_column(Integer, nullable=False)
    indice_chunk: Mapped[int] = mapped_column(Integer, nullable=False)
    conteudo: Mapped[str] = mapped_column(Text, nullable=False)
    hash_conteudo: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=False)

    __table_args__ = (
        UniqueConstraint("hash_conteudo", name="uq_documento_chunks_hash"),
        Index(
            "ix_documento_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class PostgresHybridRepository:
    def __init__(self, database_url: str) -> None:
        self._engine: AsyncEngine = create_async_engine(
            database_url, pool_pre_ping=True, connect_args={"ssl": False}
        )
        self._session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self._engine, expire_on_commit=False
        )

    async def dispose(self) -> None:
        await self._engine.dispose()

    async def inicializar_schema(self) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS unaccent;"))
            await conn.run_sync(Base.metadata.create_all)
            await conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_documento_chunks_fts_pt
                    ON documento_chunks
                    USING GIN (to_tsvector('portuguese', conteudo));
                    """
                )
            )

    async def salvar_edital(self, edital: EditalMetadata) -> None:
        valores = {
            "numero_controle_pncp": edital.numero_controle_pncp,
            "orgao_cnpj": edital.orgao_cnpj,
            "orgao_nome": edital.orgao_nome,
            "uf": edital.uf,
            "municipio": edital.municipio,
            "modalidade_nome": edital.modalidade_nome,
            "objeto_compra": edital.objeto_compra,
            "valor_total_estimado": edital.valor_total_estimado,
            "data_publicacao_pncp": edital.data_publicacao_pncp,
            "link_sistema_origem": edital.link_sistema_origem,
        }
        stmt = pg_insert(EditalORM).values(**valores)
        stmt = stmt.on_conflict_do_update(
            index_elements=["numero_controle_pncp"],
            set_={
                "objeto_compra": stmt.excluded.objeto_compra,
                "valor_total_estimado": stmt.excluded.valor_total_estimado,
                "link_sistema_origem": stmt.excluded.link_sistema_origem,
            },
        )
        async with self._session_factory() as session:
            await session.execute(stmt)
            await session.commit()

    async def salvar_chunks(
        self,
        chunks: list[DocumentoChunk],
        embeddings: list[list[float]],
    ) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("A quantidade de chunks e embeddings deve ser idêntica.")

        registros = [
            {
                "numero_controle_pncp": chunk.numero_controle_pncp,
                "arquivo_nome": chunk.arquivo_nome,
                "numero_pagina": chunk.numero_pagina,
                "indice_chunk": chunk.indice_chunk,
                "conteudo": chunk.conteudo,
                "hash_conteudo": chunk.hash_conteudo,
                "embedding": emb,
            }
            for chunk, emb in zip(chunks, embeddings, strict=True)
        ]

        stmt = (
            pg_insert(DocumentoChunkORM)
            .values(registros)
            .on_conflict_do_nothing(index_elements=["hash_conteudo"])
        )

        async with self._session_factory() as session:
            result = cast(CursorResult[Any], await session.execute(stmt))
            await session.commit()
            return int(result.rowcount or 0)

    async def busca_hibrida(
        self,
        consulta_texto: str,
        consulta_vetor: list[float],
        numero_controle_pncp: str | None = None,
        limite: int = 5,
        rrf_k: int = 60,
    ) -> list[ResultadoBuscaHibrida]:
        vetor_literal = "[" + ",".join(f"{x:.8f}" for x in consulta_vetor) + "]"
        filtro_edital = (
            "WHERE numero_controle_pncp = :numero_controle_pncp"
            if numero_controle_pncp is not None
            else ""
        )

        sql = text(
            f"""
            WITH candidatos_vetoriais AS (
                SELECT
                    id,
                    ROW_NUMBER() OVER (ORDER BY embedding <=> CAST(:query_vec AS vector)) AS rank_v
                FROM documento_chunks
                {filtro_edital}
                ORDER BY embedding <=> CAST(:query_vec AS vector)
                LIMIT :pool_size
            ),
            candidatos_lexicais AS (
                SELECT
                    id,
                    ROW_NUMBER() OVER (
                        ORDER BY ts_rank_cd(
                            to_tsvector('portuguese', conteudo),
                            plainto_tsquery('portuguese', :query_text)
                        ) DESC
                    ) AS rank_l
                FROM documento_chunks
                {filtro_edital}
                {"AND" if filtro_edital else "WHERE"} to_tsvector('portuguese', conteudo)
                    @@ plainto_tsquery('portuguese', :query_text)
                LIMIT :pool_size
            ),
            rrf_combinado AS (
                SELECT
                    COALESCE(v.id, l.id) AS chunk_id,
                    v.rank_v,
                    l.rank_l,
                    (
                        COALESCE(1.0 / (:rrf_k + v.rank_v), 0.0) +
                        COALESCE(1.0 / (:rrf_k + l.rank_l), 0.0)
                    ) AS score_rrf
                FROM candidatos_vetoriais v
                FULL OUTER JOIN candidatos_lexicais l ON v.id = l.id
            )
            SELECT
                c.numero_controle_pncp,
                c.arquivo_nome,
                c.numero_pagina,
                c.indice_chunk,
                c.conteudo,
                r.score_rrf,
                r.rank_v,
                r.rank_l
            FROM rrf_combinado r
            JOIN documento_chunks c ON c.id = r.chunk_id
            ORDER BY r.score_rrf DESC
            LIMIT :limite;
            """
        )

        params: dict[str, object] = {
            "query_vec": vetor_literal,
            "query_text": consulta_texto,
            "pool_size": max(limite * 4, 20),
            "rrf_k": rrf_k,
            "limite": limite,
        }
        if numero_controle_pncp is not None:
            params["numero_controle_pncp"] = numero_controle_pncp

        async with self._session_factory() as session:
            rows = (await session.execute(sql, params)).mappings().all()

        return [
            ResultadoBuscaHibrida(
                numero_controle_pncp=str(row["numero_controle_pncp"]),
                arquivo_nome=str(row["arquivo_nome"]),
                numero_pagina=int(row["numero_pagina"]),
                indice_chunk=int(row["indice_chunk"]),
                conteudo=str(row["conteudo"]),
                score_rrf=float(row["score_rrf"]),
                rank_vetorial=int(row["rank_v"]) if row["rank_v"] is not None else None,
                rank_lexical=int(row["rank_l"]) if row["rank_l"] is not None else None,
            )
            for row in rows
        ]
