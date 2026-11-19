from datetime import date
from decimal import Decimal

import httpx
import pytest
import respx

from licitai.adapters.pncp_client import PncpApiError, PncpHttpClient


@pytest.mark.asyncio
@respx.mock
async def test_buscar_contratacoes_sucesso() -> None:
    base_url = "https://pncp.gov.br/api/consulta/v1"
    respx.get(f"{base_url}/contratacoes/publicacao").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "numeroControlePNCP": "11222333000181-1-000001/2026",
                        "orgaoEntidade": {
                            "cnpj": "11222333000181",
                            "razaoSocial": "PREFEITURA MUNICIPAL DE SURUBIM",
                        },
                        "unidadeOrgao": {
                            "ufSigla": "PE",
                            "municipioNome": "Surubim",
                        },
                        "modalidadeNome": "Pregão - Eletrônico",
                        "objetoCompra": "Aquisição de equipamentos de informática para escolas",
                        "valorTotalEstimado": 150000.50,
                        "dataPublicacaoPncp": "2026-11-10T10:30:00",
                        "linkSistemaOrigem": "https://exemplo.gov.br/edital/1",
                    }
                ]
            },
        )
    )

    client = PncpHttpClient(base_url=base_url)
    editais = await client.buscar_contratacoes_por_data(
        data_inicial=date(2026, 11, 1),
        data_final=date(2026, 11, 15),
        uf="PE",
    )

    assert len(editais) == 1
    assert editais[0].orgao_nome == "PREFEITURA MUNICIPAL DE SURUBIM"
    assert editais[0].municipio == "Surubim"
    assert editais[0].valor_total_estimado == Decimal("150000.5")


@pytest.mark.asyncio
@respx.mock
async def test_buscar_contratacoes_sem_conteudo_204() -> None:
    base_url = "https://pncp.gov.br/api/consulta/v1"
    respx.get(f"{base_url}/contratacoes/publicacao").mock(return_value=httpx.Response(204))

    client = PncpHttpClient(base_url=base_url)
    editais = await client.buscar_contratacoes_por_data(
        data_inicial=date(2026, 11, 1),
        data_final=date(2026, 11, 2),
        uf="PE",
    )

    assert editais == []


@pytest.mark.asyncio
@respx.mock
async def test_buscar_contratacoes_erro_servidor_dispara_excecao() -> None:
    base_url = "https://pncp.gov.br/api/consulta/v1"
    respx.get(f"{base_url}/contratacoes/publicacao").mock(return_value=httpx.Response(503))

    client = PncpHttpClient(base_url=base_url)
    with pytest.raises(PncpApiError):
        await client.buscar_contratacoes_por_data(
            data_inicial=date(2026, 11, 1),
            data_final=date(2026, 11, 2),
            uf="PE",
        )
