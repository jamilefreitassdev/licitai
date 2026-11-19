import logging
from dataclasses import dataclass

from licitai.adapters.pdf_parser import PyMuPdfParser
from licitai.domain.models import EditalMetadata
from licitai.domain.ports import EditalRepository, EmbeddingProvider, PncpGateway

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestaoResultado:
    numero_controle_pncp: str
    chunks_extraidos: int
    chunks_inseridos: int


class EditalIngestionService:
    def __init__(
        self,
        pncp_gateway: PncpGateway,
        pdf_parser: PyMuPdfParser,
        embedding_provider: EmbeddingProvider,
        repository: EditalRepository,
    ) -> None:
        self._pncp = pncp_gateway
        self._parser = pdf_parser
        self._embedder = embedding_provider
        self._repo = repository

    @staticmethod
    def _extrair_chave_pncp(numero_controle_pncp: str) -> tuple[str, int, int]:
        # Formato padrão PNCP: "{cnpj}-{esfera}-{sequencial}/{ano}"
        parte_principal, ano_str = numero_controle_pncp.split("/")
        partes = parte_principal.split("-")
        cnpj = partes[0]
        sequencial = int(partes[2])
        return cnpj, int(ano_str), sequencial

    async def ingerir_edital_com_pdf(
        self,
        edital: EditalMetadata,
        arquivo_nome: str = "edital.pdf",
        id_documento: int = 1,
    ) -> IngestaoResultado:
        cnpj, ano, sequencial = self._extrair_chave_pncp(edital.numero_controle_pncp)

        pdf_bytes = await self._pncp.baixar_documento_pdf(
            cnpj=cnpj,
            ano=ano,
            sequencial=sequencial,
            id_documento=id_documento,
        )

        chunks = self._parser.extrair_chunks(
            pdf_bytes=pdf_bytes,
            numero_controle_pncp=edital.numero_controle_pncp,
            arquivo_nome=arquivo_nome,
        )

        await self._repo.salvar_edital(edital)

        if not chunks:
            return IngestaoResultado(
                numero_controle_pncp=edital.numero_controle_pncp,
                chunks_extraidos=0,
                chunks_inseridos=0,
            )

        vetores = await self._embedder.embed_texts([c.conteudo for c in chunks])
        inseridos = await self._repo.salvar_chunks(chunks=chunks, embeddings=vetores)

        logger.info(
            "Ingestão concluída para %s: %d extraídos, %d novos inseridos.",
            edital.numero_controle_pncp,
            len(chunks),
            inseridos,
        )

        return IngestaoResultado(
            numero_controle_pncp=edital.numero_controle_pncp,
            chunks_extraidos=len(chunks),
            chunks_inseridos=inseridos,
        )
