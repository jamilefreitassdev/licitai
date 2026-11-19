import pymupdf
import pytest

from licitai.adapters.pdf_parser import PdfParsingError, PyMuPdfParser


def _criar_pdf_em_memoria(paginas: list[str]) -> bytes:
    doc = pymupdf.open()
    area_texto = pymupdf.Rect(50, 50, 545, 790)
    for texto in paginas:
        pagina = doc.new_page()
        pagina.insert_textbox(area_texto, texto)
    pdf_bytes: bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_extrair_chunks_preserva_numero_pagina_e_hash() -> None:
    pdf_bytes = _criar_pdf_em_memoria(
        [
            "EDITAL DE PREGÃO ELETRÔNICO Nº 010/2026. Objeto: Aquisição de servidores.",
            "CLÁUSULA QUINTA: O prazo máximo de entrega será de 15 dias úteis após empenho.",
        ]
    )

    parser = PyMuPdfParser(chunk_size=500, chunk_overlap=50)
    chunks = parser.extrair_chunks(
        pdf_bytes=pdf_bytes,
        numero_controle_pncp="11222333000181-1-000010/2026",
        arquivo_nome="edital_010_2026.pdf",
    )

    assert len(chunks) == 2
    assert chunks[0].numero_pagina == 1
    assert "Aquisição de servidores" in chunks[0].conteudo
    assert chunks[1].numero_pagina == 2
    assert "15 dias úteis" in chunks[1].conteudo
    assert len(chunks[0].hash_conteudo) == 64
    assert chunks[0].hash_conteudo != chunks[1].hash_conteudo


def test_extrair_chunks_fatiamento_com_overlap() -> None:
    texto_longo = "Artigo 1º. " + ("Condição técnica obrigatória do edital. " * 25)
    pdf_bytes = _criar_pdf_em_memoria([texto_longo])

    parser = PyMuPdfParser(chunk_size=200, chunk_overlap=40)
    chunks = parser.extrair_chunks(
        pdf_bytes=pdf_bytes,
        numero_controle_pncp="11222333000181-1-000011/2026",
        arquivo_nome="anexo_tecnico.pdf",
    )

    assert len(chunks) >= 4
    assert all(c.numero_pagina == 1 for c in chunks)
    assert [c.indice_chunk for c in chunks] == list(range(len(chunks)))


def test_extrair_chunks_pdf_invalido_dispara_erro() -> None:
    parser = PyMuPdfParser()
    with pytest.raises(PdfParsingError):
        parser.extrair_chunks(
            pdf_bytes=b"isso-nao-eh-um-pdf",
            numero_controle_pncp="11222333000181-1-000012/2026",
            arquivo_nome="corrompido.pdf",
        )
