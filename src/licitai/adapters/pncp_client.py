import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from licitai.domain.models import EditalMetadata

logger = logging.getLogger(__name__)


class PncpApiError(Exception):
    """Erro de comunicação ou resposta inválida da API do PNCP."""


class PncpHttpClient:
    def __init__(self, base_url: str, timeout_seconds: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout_seconds)

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=8),
        reraise=True,
    )
    async def _get(self, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, params=params)
            if response.status_code in {429, 500, 502, 503, 504}:
                response.raise_for_status()
            return response

    async def buscar_contratacoes_por_data(
        self,
        data_inicial: date,
        data_final: date,
        codigo_modalidade: int = 6,
        uf: str = "PE",
        pagina: int = 1,
        tamanho_pagina: int = 20,
    ) -> list[EditalMetadata]:
        url = f"{self._base_url}/contratacoes/publicacao"
        params: dict[str, Any] = {
            "dataInicial": data_inicial.strftime("%Y%m%d"),
            "dataFinal": data_final.strftime("%Y%m%d"),
            "codigoModalidadeContratacao": codigo_modalidade,
            "uf": uf.upper(),
            "pagina": pagina,
            "tamanhoPagina": tamanho_pagina,
        }

        try:
            response = await self._get(url, params=params)
        except httpx.HTTPError as exc:
            logger.error("Falha de rede ao consultar PNCP: %s", exc)
            raise PncpApiError(f"Erro ao consultar PNCP: {exc}") from exc

        if response.status_code == 204:
            return []

        if response.status_code != 200:
            raise PncpApiError(f"Status inesperado do PNCP: {response.status_code}")

        payload = response.json()
        itens = payload.get("data", [])
        resultados: list[EditalMetadata] = []

        for item in itens:
            try:
                orgao = item.get("orgaoEntidade", {})
                unidade = item.get("unidadeOrgao", {})
                valor = item.get("valorTotalEstimado")

                edital = EditalMetadata(
                    numero_controle_pncp=item["numeroControlePNCP"],
                    orgao_cnpj=orgao["cnpj"],
                    orgao_nome=orgao["razaoSocial"],
                    uf=unidade.get("ufSigla", uf.upper()),
                    municipio=unidade.get("municipioNome"),
                    modalidade_nome=item.get("modalidadeNome", "Não informada"),
                    objeto_compra=item["objetoCompra"],
                    valor_total_estimado=Decimal(str(valor)) if valor is not None else None,
                    data_publicacao_pncp=datetime.fromisoformat(item["dataPublicacaoPncp"]),
                    link_sistema_origem=item.get("linkSistemaOrigem"),
                )
                resultados.append(edital)
            except (KeyError, ValueError) as exc:
                logger.warning("Registro do PNCP ignorado por schema inválido: %s", exc)
                continue

        return resultados

    async def baixar_documento_pdf(
        self,
        cnpj: str,
        ano: int,
        sequencial: int,
        id_documento: int = 1,
    ) -> bytes:
        pncp_files_base = self._base_url.replace("/api/consulta/v1", "/api/pncp/v1")
        url = f"{pncp_files_base}/orgaos/{cnpj}/compras/{ano}/{sequencial}/arquivos/{id_documento}"

        try:
            response = await self._get(url)
        except httpx.HTTPError as exc:
            raise PncpApiError(f"Falha ao baixar arquivo do PNCP: {exc}") from exc

        if response.status_code != 200:
            raise PncpApiError(f"Arquivo não encontrado no PNCP (HTTP {response.status_code})")

        return response.content
