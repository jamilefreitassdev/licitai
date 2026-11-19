import re

from licitai.domain.models import CitacaoDocumental, ParecerAuditoria, ResultadoBuscaHibrida
from licitai.domain.ports import EditalRepository, EmbeddingProvider, LlmGenerator


class EditalAuditService:
    def __init__(
        self,
        repository: EditalRepository,
        embedding_provider: EmbeddingProvider,
        generator: LlmGenerator,
        min_rrf_threshold: float = 0.015,
    ) -> None:
        self._repo = repository
        self._embedder = embedding_provider
        self._generator = generator
        self._min_rrf_threshold = min_rrf_threshold

    @staticmethod
    def _calcular_fidelidade_lexical(
        resposta: str,
        evidencias: list[ResultadoBuscaHibrida],
    ) -> float:
        if not evidencias:
            return 0.0

        tokens_contexto = set(
            re.findall(r"\b[a-zA-ZÀ-ÿ0-9]{4,}\b", " ".join(e.conteudo for e in evidencias).lower())
        )
        tokens_resposta = [
            t
            for t in re.findall(r"\b[a-zA-ZÀ-ÿ0-9]{4,}\b", resposta.lower())
            if t not in {"base", "evidências", "auditadas", "para"}
        ]
        if not tokens_resposta:
            return 0.0

        ancorados = sum(1 for t in tokens_resposta if t in tokens_contexto)
        return round(min(ancorados / len(tokens_resposta), 1.0), 4)

    async def auditar_pergunta(
        self,
        pergunta: str,
        numero_controle_pncp: str | None = None,
        limite_evidencias: int = 5,
    ) -> ParecerAuditoria:
        query_vec = (await self._embedder.embed_texts([pergunta]))[0]
        resultados = await self._repo.busca_hibrida(
            consulta_texto=pergunta,
            consulta_vetor=query_vec,
            numero_controle_pncp=numero_controle_pncp,
            limite=limite_evidencias,
        )

        evidencias_validas = [r for r in resultados if r.score_rrf >= self._min_rrf_threshold]

        if not evidencias_validas:
            return ParecerAuditoria(
                pergunta=pergunta,
                resposta_fundamentada=(
                    "Evidência insuficiente nos editais indexados para emitir parecer seguro."
                ),
                citacoes=[],
                indice_fidelidade=0.0,
                evidencia_suficiente=False,
            )

        resposta = await self._generator.gerar_parecer(
            pergunta=pergunta,
            evidencias=evidencias_validas,
        )
        fidelidade = self._calcular_fidelidade_lexical(resposta, evidencias_validas)

        citacoes = [
            CitacaoDocumental(
                numero_controle_pncp=item.numero_controle_pncp,
                arquivo_nome=item.arquivo_nome,
                numero_pagina=item.numero_pagina,
                trecho_original=item.conteudo,
                score_rrf=round(item.score_rrf, 6),
            )
            for item in evidencias_validas
        ]

        return ParecerAuditoria(
            pergunta=pergunta,
            resposta_fundamentada=resposta,
            citacoes=citacoes,
            indice_fidelidade=fidelidade,
            evidencia_suficiente=True,
        )
