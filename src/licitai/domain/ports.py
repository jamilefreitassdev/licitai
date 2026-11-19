from datetime import date
from typing import Protocol

from licitai.domain.models import EditalMetadata


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
