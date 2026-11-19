from fastapi.testclient import TestClient

from licitai.adapters.embeddings import DeterministicHashEmbeddingProvider
from licitai.adapters.llm_client import ExtractiveGroundedGenerator
from licitai.api.main import create_app
from licitai.domain.models import DocumentoChunk, EditalMetadata, ResultadoBuscaHibrida
from licitai.services.audit_service import EditalAuditService


class StubHybridRepo:
    def __init__(self, resultados: list[ResultadoBuscaHibrida]) -> None:
        self._resultados = resultados

    async def inicializar_schema(self) -> None:
        pass

    async def salvar_edital(self, edital: EditalMetadata) -> None:
        pass

    async def salvar_chunks(
        self,
        chunks: list[DocumentoChunk],
        embeddings: list[list[float]],
    ) -> int:
        return len(chunks)

    async def busca_hibrida(
        self,
        consulta_texto: str,
        consulta_vetor: list[float],
        numero_controle_pncp: str | None = None,
        limite: int = 5,
        rrf_k: int = 60,
    ) -> list[ResultadoBuscaHibrida]:
        return self._resultados


def test_health_e_auditoria_com_citacoes() -> None:
    app = create_app()
    evidencia = ResultadoBuscaHibrida(
        numero_controle_pncp="11222333000181-1-000050/2026",
        arquivo_nome="edital_merenda.pdf",
        numero_pagina=12,
        indice_chunk=1,
        conteudo="O prazo de entrega dos gêneros alimentícios será de até 5 dias úteis.",
        score_rrf=0.032787,
        rank_vetorial=1,
        rank_lexical=1,
    )
    app.state.audit_service = EditalAuditService(
        repository=StubHybridRepo([evidencia]),
        embedding_provider=DeterministicHashEmbeddingProvider(dimension=384),
        generator=ExtractiveGroundedGenerator(),
    )

    client = TestClient(app, raise_server_exceptions=True)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    resp = client.post(
        "/api/v1/auditoria/perguntar",
        json={"pergunta": "Qual o prazo de entrega dos gêneros alimentícios?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["evidencia_suficiente"] is True
    assert body["indice_fidelidade"] > 0.5
    assert len(body["citacoes"]) == 1
    assert body["citacoes"][0]["numero_pagina"] == 12


def test_auditoria_sem_evidencia_retorna_recusa_segura() -> None:
    app = create_app()
    app.state.audit_service = EditalAuditService(
        repository=StubHybridRepo([]),
        embedding_provider=DeterministicHashEmbeddingProvider(dimension=384),
        generator=ExtractiveGroundedGenerator(),
    )

    client = TestClient(app)
    resp = client.post(
        "/api/v1/auditoria/perguntar",
        json={"pergunta": "Qual o valor da garantia bancária exigida?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["evidencia_suficiente"] is False
    assert body["citacoes"] == []
    assert body["indice_fidelidade"] == 0.0
