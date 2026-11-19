from datetime import date
from typing import Protocol

from licitai.domain.models import DocumentoChunk, EditalMetadata, ResultadoBuscaHibrida


class PncpGateway(Protocol):
    async def buscar_contratacoes_por_data(
        self,
        data_inicial: date,
        data_final: date,
        codigo_modalidade: int = 6,
        uf: str = "PE",
        pagina: int = 1,
        tamanho_pagina: int = 20,
    ) -> list[EditalMetadata]:
        """Consulta contratações publicadas no PNCP filtradas por período e UF."""
        ...

    async def baixar_documento_pdf(
        self,
        cnpj: str,
        ano: int,
        sequencial: int,
        id_documento: int = 1,
    ) -> bytes:
        """Faz o download dos bytes brutos de um documento anexo ao edital."""
        ...


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int:
        """Dimensão do vetor de saída."""
        ...

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Gera embeddings normalizados (L2) para uma lista de textos."""
        ...


class EditalRepository(Protocol):
    async def inicializar_schema(self) -> None:
        """Garante extensões, tabelas e índices no PostgreSQL."""
        ...

    async def salvar_edital(self, edital: EditalMetadata) -> None:
        """Insere ou atualiza os metadados de um edital (idempotente)."""
        ...

    async def salvar_chunks(
        self,
        chunks: list[DocumentoChunk],
        embeddings: list[list[float]],
    ) -> int:
        """Persiste chunks com seus vetores ignorando duplicatas por SHA-256."""
        ...

    async def busca_hibrida(
        self,
        consulta_texto: str,
        consulta_vetor: list[float],
        numero_controle_pncp: str | None = None,
        limite: int = 5,
        rrf_k: int = 60,
    ) -> list[ResultadoBuscaHibrida]:
        """Executa busca híbrida (pgvector + tsvector) ranqueada por RRF."""
        ...


class LlmGenerator(Protocol):
    async def gerar_parecer(
        self,
        pergunta: str,
        evidencias: list[ResultadoBuscaHibrida],
    ) -> str:
        """Sintetiza uma resposta estritamente fundamentada nas evidências fornecidas."""
        ...
