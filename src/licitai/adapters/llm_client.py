from licitai.domain.models import ResultadoBuscaHibrida


class ExtractiveGroundedGenerator:
    """Sintetizador determinístico estritamente ancorado nas evidências recuperadas."""

    async def gerar_parecer(
        self,
        pergunta: str,
        evidencias: list[ResultadoBuscaHibrida],
    ) -> str:
        if not evidencias:
            return (
                "Não foram encontradas cláusulas nos documentos indexados que respondam "
                f"com segurança à consulta: '{pergunta}'."
            )

        linhas = [f"Com base nas evidências auditadas para '{pergunta}':"]
        for item in evidencias[:3]:
            linhas.append(f'- [{item.arquivo_nome}, pág. {item.numero_pagina}]: "{item.conteudo}"')
        return "\n".join(linhas)
