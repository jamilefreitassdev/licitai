import socket
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from licitai.adapters.embeddings import DeterministicHashEmbeddingProvider
from licitai.adapters.pgvector_repo import PostgresHybridRepository
from licitai.core.config import get_settings
from licitai.domain.models import DocumentoChunk, EditalMetadata


def _postgres_disponivel() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 5433), timeout=1.0):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not _postgres_disponivel(), reason="Container PostgreSQL indisponível")
@pytest.mark.asyncio
async def test_persistencia_e_busca_hibrida_rrf_postgresql() -> None:
    settings = get_settings()
    repo = PostgresHybridRepository(str(settings.database_url))
    embedder = DeterministicHashEmbeddingProvider(dimension=384)

    try:
        await repo.inicializar_schema()

        edital = EditalMetadata(
            numero_controle_pncp="99888777000166-1-000099/2026",
            orgao_cnpj="99888777000166",
            orgao_nome="FUNDO MUNICIPAL DE SAÚDE DE SURUBIM",
            uf="PE",
            municipio="Surubim",
            modalidade_nome="Pregão - Eletrônico",
            objeto_compra="Contratação de sistema de prontuário eletrônico",
            valor_total_estimado=Decimal("420000.00"),
            data_publicacao_pncp=datetime(2026, 11, 18, 12, 0, tzinfo=UTC),
        )
        await repo.salvar_edital(edital)

        chunks = [
            DocumentoChunk(
                numero_controle_pncp=edital.numero_controle_pncp,
                arquivo_nome="edital_saude.pdf",
                numero_pagina=1,
                indice_chunk=0,
                conteudo="DA HABILITAÇÃO JURÍDICA: Apresentar contrato social consolidado.",
                hash_conteudo="a" * 64,
            ),
            DocumentoChunk(
                numero_controle_pncp=edital.numero_controle_pncp,
                arquivo_nome="edital_saude.pdf",
                numero_pagina=7,
                indice_chunk=0,
                conteudo="DA MULTA CONTRATUAL: O atraso injustificado sujeitará multa de 10%.",
                hash_conteudo="b" * 64,
            ),
        ]
        vetores = await embedder.embed_texts([c.conteudo for c in chunks])
        await repo.salvar_chunks(chunks, vetores)

        pergunta = "Qual o percentual de multa por atraso contratual?"
        q_vec = (await embedder.embed_texts([pergunta]))[0]
        resultados = await repo.busca_hibrida(
            consulta_texto=pergunta,
            consulta_vetor=q_vec,
            numero_controle_pncp=edital.numero_controle_pncp,
            limite=2,
        )

        assert len(resultados) >= 1
        assert resultados[0].numero_pagina == 7
        assert "multa de 10%" in resultados[0].conteudo
        assert resultados[0].score_rrf > 0.0
    finally:
        await repo.dispose()
