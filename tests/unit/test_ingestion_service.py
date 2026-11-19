from datetime import UTC, date, datetime
from decimal import Decimal

import pymupdf
import pytest

from licitai.adapters.embeddings import DeterministicHashEmbeddingProvider
from licitai.adapters.pdf_parser import PyMuPdfParser
from licitai.domain.models import DocumentoChunk, EditalMetadata, ResultadoBuscaHibrida
from licitai.services.ingestion_service import EditalIngestionService


class FakePncpGateway:
    def __init__(self, pdf_bytes: bytes) -> None:
        self._pdf_bytes = pdf_bytes

    async def buscar_contratacoes_por_data(
        self,
        data_inicial: date,
        data_final: date,
        codigo_modalidade: int = 6,
        uf: str = "PE",
        pagina: int = 1,
        tamanho_pagina: int = 20,
    ) -> list[EditalMetadata]:
        return []

    async def baixar_documento_pdf(
        self,
        cnpj: str,
        ano: int,
        sequencial: int,
        id_documento: int = 1,
    ) -> bytes:
        return self._pdf_bytes


class InMemoryEditalRepository:
    def __init__(self) -> None:
        self.editais: dict[str, EditalMetadata] = {}
        self.chunks_por_hash: dict[str, DocumentoChunk] = {}

    async def inicializar_schema(self) -> None:
        pass

    async def salvar_edital(self, edital: EditalMetadata) -> None:
        self.editais[edital.numero_controle_pncp] = edital

    async def salvar_chunks(
        self,
        chunks: list[DocumentoChunk],
        embeddings: list[list[float]],
    ) -> int:
        novos = 0
        for chunk in chunks:
            if chunk.hash_conteudo not in self.chunks_por_hash:
                self.chunks_por_hash[chunk.hash_conteudo] = chunk
                novos += 1
        return novos

    async def busca_hibrida(
        self,
        consulta_texto: str,
        consulta_vetor: list[float],
        numero_controle_pncp: str | None = None,
        limite: int = 5,
        rrf_k: int = 60,
    ) -> list[ResultadoBuscaHibrida]:
        return []


@pytest.mark.asyncio
async def test_ingestao_edital_end_to_end_e_idempotencia() -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(
        pymupdf.Rect(50, 50, 545, 790),
        "PREGÃO ELETRÔNICO 042/2026. Garantia mínima de 36 meses on-site.",
    )
    pdf_bytes: bytes = doc.tobytes()
    doc.close()

    edital = EditalMetadata(
        numero_controle_pncp="11222333000181-1-000042/2026",
        orgao_cnpj="11222333000181",
        orgao_nome="MUNICÍPIO DE CASINHAS",
        uf="PE",
        municipio="Casinhas",
        modalidade_nome="Pregão - Eletrônico",
        objeto_compra="Aquisição de notebooks educacionais",
        valor_total_estimado=Decimal("280000.00"),
        data_publicacao_pncp=datetime(2026, 11, 18, 9, 0, tzinfo=UTC),
    )

    repo = InMemoryEditalRepository()
    service = EditalIngestionService(
        pncp_gateway=FakePncpGateway(pdf_bytes),
        pdf_parser=PyMuPdfParser(),
        embedding_provider=DeterministicHashEmbeddingProvider(dimension=384),
        repository=repo,
    )

    res1 = await service.ingerir_edital_com_pdf(edital)
    assert res1.chunks_extraidos == 1
    assert res1.chunks_inseridos == 1

    # Segunda execução com o mesmo PDF deve ser idempotente (0 novos inseridos)
    res2 = await service.ingerir_edital_com_pdf(edital)
    assert res2.chunks_extraidos == 1
    assert res2.chunks_inseridos == 0
