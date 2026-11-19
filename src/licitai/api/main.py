from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from licitai.adapters.embeddings import DeterministicHashEmbeddingProvider
from licitai.adapters.llm_client import ExtractiveGroundedGenerator
from licitai.adapters.pdf_parser import PyMuPdfParser
from licitai.adapters.pgvector_repo import PostgresHybridRepository
from licitai.adapters.pncp_client import PncpHttpClient
from licitai.api.v1.endpoints import router as v1_router
from licitai.core.config import get_settings
from licitai.core.logging import configure_logging
from licitai.services.audit_service import EditalAuditService
from licitai.services.ingestion_service import EditalIngestionService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)

    repo = PostgresHybridRepository(str(settings.database_url))
    await repo.inicializar_schema()

    pncp_client = PncpHttpClient(
        base_url=settings.pncp_base_url,
        timeout_seconds=settings.pncp_timeout_seconds,
    )
    embedder = DeterministicHashEmbeddingProvider(dimension=384)
    parser = PyMuPdfParser()
    generator = ExtractiveGroundedGenerator()

    app.state.repo = repo
    app.state.pncp_client = pncp_client
    app.state.ingestion_service = EditalIngestionService(
        pncp_gateway=pncp_client,
        pdf_parser=parser,
        embedding_provider=embedder,
        repository=repo,
    )
    app.state.audit_service = EditalAuditService(
        repository=repo,
        embedding_provider=embedder,
        generator=generator,
    )

    try:
        yield
    finally:
        await repo.dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Auditor Inteligente de Editais e Contratos Públicos com RAG Avaliado",
        lifespan=lifespan,
    )

    @app.get("/health", tags=["Observabilidade"])
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok", "service": settings.app_name, "version": settings.app_version}

    app.include_router(v1_router, prefix="/api/v1")
    return app


app = create_app()
