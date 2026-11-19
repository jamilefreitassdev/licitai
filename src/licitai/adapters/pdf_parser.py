import hashlib
import re

import pymupdf

from licitai.domain.models import DocumentoChunk


class PdfParsingError(Exception):
    """Erro ao abrir ou extrair conteúdo de um arquivo PDF."""


class PyMuPdfParser:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap deve ser estritamente menor que chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @staticmethod
    def _normalizar_texto(texto: str) -> str:
        # Preserva quebras de parágrafo, mas limpa espaços e linhas em branco excessivas
        texto = re.sub(r"[ \t]+", " ", texto)
        texto = re.sub(r"\n{3,}", "\n\n", texto)
        return texto.strip()

    def _fatiar_texto(self, texto: str) -> list[str]:
        if len(texto) <= self._chunk_size:
            return [texto] if len(texto) >= 10 else []

        chunks: list[str] = []
        inicio = 0
        tamanho_total = len(texto)

        while inicio < tamanho_total:
            fim = min(inicio + self._chunk_size, tamanho_total)

            # Recua até uma quebra de parágrafo ou fim de frase para não cortar palavras
            if fim < tamanho_total:
                quebra_paragrafo = texto.rfind("\n\n", inicio + self._chunk_overlap, fim)
                quebra_ponto = texto.rfind(". ", inicio + self._chunk_overlap, fim)
                melhor_corte = max(quebra_paragrafo, quebra_ponto)
                if melhor_corte != -1:
                    fim = melhor_corte + 1

            trecho = texto[inicio:fim].strip()
            if len(trecho) >= 10:
                chunks.append(trecho)

            if fim >= tamanho_total:
                break

            inicio = max(fim - self._chunk_overlap, inicio + 1)

        return chunks

    def extrair_chunks(
        self,
        pdf_bytes: bytes,
        numero_controle_pncp: str,
        arquivo_nome: str,
    ) -> list[DocumentoChunk]:
        if not pdf_bytes:
            raise PdfParsingError("O buffer de bytes do PDF está vazio.")

        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            raise PdfParsingError(f"Falha ao abrir PDF '{arquivo_nome}': {exc}") from exc

        resultados: list[DocumentoChunk] = []

        with doc:
            for idx_pagina in range(len(doc)):
                numero_pagina = idx_pagina + 1
                texto_bruto = str(doc[idx_pagina].get_text("text"))
                texto_limpo = self._normalizar_texto(texto_bruto)

                if len(texto_limpo) < 10:
                    continue

                trechos = self._fatiar_texto(texto_limpo)
                for idx_chunk, conteudo in enumerate(trechos):
                    assinatura = (
                        f"{numero_controle_pncp}:{arquivo_nome}:"
                        f"{numero_pagina}:{idx_chunk}:{conteudo}"
                    )
                    hash_sha256 = hashlib.sha256(assinatura.encode("utf-8")).hexdigest()

                    resultados.append(
                        DocumentoChunk(
                            numero_controle_pncp=numero_controle_pncp,
                            arquivo_nome=arquivo_nome,
                            numero_pagina=numero_pagina,
                            indice_chunk=idx_chunk,
                            conteudo=conteudo,
                            hash_conteudo=hash_sha256,
                        )
                    )

        return resultados
