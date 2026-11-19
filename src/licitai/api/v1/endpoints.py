from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from licitai.adapters.pncp_client import PncpApiError, PncpHttpClient
from licitai.domain.models import EditalMetadata, ParecerAuditoria
from licitai.services.audit_service import EditalAuditService
from licitai.services.ingestion_service import EditalIngestionService

router = APIRouter()


class IngestaoRequest(BaseModel):
    edital: EditalMetadata
    arquivo_nome: str = Field(default="edital.pdf", min_length=3)
    id_documento: int = Field(default=1, ge=1)


class IngestaoResponse(BaseModel):
    numero_controle_pncp: str
    chunks_extraidos: int
    chunks_inseridos: int


class PerguntaAuditoriaRequest(BaseModel):
    pergunta: str = Field(..., min_length=5, max_length=1000)
    numero_controle_pncp: str | None = None
    limite_evidencias: int = Field(default=5, ge=1, le=20)


def get_pncp_client(request: Request) -> PncpHttpClient:
    client: PncpHttpClient = request.app.state.pncp_client
    return client


def get_ingestion_service(request: Request) -> EditalIngestionService:
    service: EditalIngestionService = request.app.state.ingestion_service
    return service


def get_audit_service(request: Request) -> EditalAuditService:
    service: EditalAuditService = request.app.state.audit_service
    return service


@router.get("/editais/pncp", response_model=list[EditalMetadata], tags=["PNCP"])
async def listar_editais_pncp(
    pncp_client: Annotated[PncpHttpClient, Depends(get_pncp_client)],
    data_inicial: Annotated[date, Query(description="Data inicial de publicação (YYYY-MM-DD)")],
    data_final: Annotated[date, Query(description="Data final de publicação (YYYY-MM-DD)")],
    uf: Annotated[str, Query(min_length=2, max_length=2)] = "PE",
    codigo_modalidade: Annotated[int, Query(description="6 = Pregão Eletrônico")] = 6,
    pagina: Annotated[int, Query(ge=1)] = 1,
    tamanho_pagina: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[EditalMetadata]:
    try:
        return await pncp_client.buscar_contratacoes_por_data(
            data_inicial=data_inicial,
            data_final=data_final,
            codigo_modalidade=codigo_modalidade,
            uf=uf,
            pagina=pagina,
            tamanho_pagina=tamanho_pagina,
        )
    except PncpApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/editais/ingerir", response_model=IngestaoResponse, tags=["Ingestão"])
async def ingerir_edital(
    payload: IngestaoRequest,
    ingestion_service: Annotated[EditalIngestionService, Depends(get_ingestion_service)],
) -> IngestaoResponse:
    try:
        resultado = await ingestion_service.ingerir_edital_com_pdf(
            edital=payload.edital,
            arquivo_nome=payload.arquivo_nome,
            id_documento=payload.id_documento,
        )
        return IngestaoResponse(
            numero_controle_pncp=resultado.numero_controle_pncp,
            chunks_extraidos=resultado.chunks_extraidos,
            chunks_inseridos=resultado.chunks_inseridos,
        )
    except PncpApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/auditoria/perguntar", response_model=ParecerAuditoria, tags=["Auditoria RAG"])
async def auditar_pergunta(
    payload: PerguntaAuditoriaRequest,
    audit_service: Annotated[EditalAuditService, Depends(get_audit_service)],
) -> ParecerAuditoria:
    return await audit_service.auditar_pergunta(
        pergunta=payload.pergunta,
        numero_controle_pncp=payload.numero_controle_pncp,
        limite_evidencias=payload.limite_evidencias,
    )
