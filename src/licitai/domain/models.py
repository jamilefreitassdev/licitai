from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class EditalMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    numero_controle_pncp: str = Field(
        ..., min_length=5, description="ID único da contratação no PNCP"
    )
    orgao_cnpj: str = Field(..., pattern=r"^\d{14}$")
    orgao_nome: str = Field(..., min_length=2)
    uf: str = Field(..., min_length=2, max_length=2)
    municipio: str | None = None
    modalidade_nome: str
    objeto_compra: str = Field(..., min_length=3)
    valor_total_estimado: Decimal | None = None
    data_publicacao_pncp: datetime
    link_sistema_origem: str | None = None


class DocumentoChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    numero_controle_pncp: str
    arquivo_nome: str
    numero_pagina: int = Field(..., ge=1)
    indice_chunk: int = Field(..., ge=0)
    conteudo: str = Field(..., min_length=10)
    hash_conteudo: str = Field(..., min_length=64, max_length=64)


class ResultadoBuscaHibrida(BaseModel):
    model_config = ConfigDict(frozen=True)

    numero_controle_pncp: str
    arquivo_nome: str
    numero_pagina: int
    indice_chunk: int
    conteudo: str
    score_rrf: float
    rank_vetorial: int | None = None
    rank_lexical: int | None = None


class CitacaoDocumental(BaseModel):
    model_config = ConfigDict(frozen=True)

    numero_controle_pncp: str
    arquivo_nome: str
    numero_pagina: int
    trecho_original: str
    score_rrf: float


class ParecerAuditoria(BaseModel):
    model_config = ConfigDict(frozen=True)

    pergunta: str
    resposta_fundamentada: str
    citacoes: list[CitacaoDocumental]
    indice_fidelidade: float = Field(..., ge=0.0, le=1.0)
    evidencia_suficiente: bool
