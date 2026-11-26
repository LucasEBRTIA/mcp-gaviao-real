"""Shared synthetic fixtures for mcp-gaviao-real test suite."""

from __future__ import annotations

import io
from pathlib import Path

import docx
import openpyxl
import pymupdf
import pytest
from PIL import Image, ImageDraw

from mcp_gaviao_real.config import GaviaoSettings
from mcp_gaviao_real.router import DocumentRouter


def _make_logo_png() -> bytes:
    """Create a 120x100 colorful header logo PNG."""
    img = Image.new("RGB", (120, 100), (245, 245, 250))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 110, 90], fill=(20, 60, 140), outline=(220, 180, 40), width=4)
    draw.ellipse([35, 25, 85, 75], fill=(240, 190, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_chart_png() -> bytes:
    """Create a distinct 320x220 figure PNG."""
    img = Image.new("RGB", (320, 220), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.line([(40, 180), (300, 180)], fill=(0, 0, 0), width=3)
    draw.line([(40, 20), (40, 180)], fill=(0, 0, 0), width=3)
    heights = [40, 80, 130, 105, 150]
    for i, h in enumerate(heights):
        x0 = 60 + i * 45
        draw.rectangle([x0, 180 - h, x0 + 30, 180], fill=(30 + i * 35, 90, 200 - i * 20))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_scanned_page_png() -> bytes:
    """Create a full-page raster scan simulation (595x842)."""
    img = Image.new("RGB", (595, 842), (252, 250, 245))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 555, 800], outline=(80, 80, 80), width=2)
    for y in range(90, 700, 28):
        draw.line([(70, y), (520, y)], fill=(40, 40, 40), width=3)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def isolated_settings(tmp_path: Path) -> GaviaoSettings:
    return GaviaoSettings(
        cache_dir=tmp_path / "cache",
        max_cache_bytes=50 * 1024 * 1024,
    )


@pytest.fixture()
def router(isolated_settings: GaviaoSettings) -> DocumentRouter:
    r = DocumentRouter(isolated_settings)
    yield r
    r.close()


@pytest.fixture()
def mixed_pdf_path(tmp_path: Path) -> Path:
    """Create a 6-page synthetic PDF covering clean text, TOC, logos, scan, garbled text, and vector chart."""
    pdf_path = tmp_path / "relatorio_misto.pdf"
    logo_bytes = _make_logo_png()
    chart_bytes = _make_chart_png()
    scan_bytes = _make_scanned_page_png()

    doc = pymupdf.open()

    # Page 1: Clean Portuguese digital text + header logo
    p1 = doc.new_page(width=595, height=842)
    p1.insert_image(pymupdf.Rect(420, 20, 540, 120), stream=logo_bytes)
    p1.insert_text((50, 80), "1. Sumário Executivo", fontsize=18)
    p1.insert_text(
        (50, 130),
        "O Relatório Líquido do exercício fiscal apresenta crescimento consolidado de 18,5%.\n"
        "A receita operacional manteve estabilidade ao longo dos quatro trimestres.",
        fontsize=11,
    )

    # Page 2: Clean digital text + distinct chart figure + header logo
    p2 = doc.new_page(width=595, height=842)
    p2.insert_image(pymupdf.Rect(420, 20, 540, 120), stream=logo_bytes)
    p2.insert_text((50, 80), "2. Análise Financeira", fontsize=18)
    p2.insert_text(
        (50, 125),
        "Abaixo demonstramos a evolução trimestral das margens e custos operacionais.",
        fontsize=11,
    )
    p2.insert_image(pymupdf.Rect(80, 170, 440, 410), stream=chart_bytes)

    # Page 3: Scanned page (full-page image, no text layer)
    p3 = doc.new_page(width=595, height=842)
    p3.insert_image(pymupdf.Rect(0, 0, 595, 842), stream=scan_bytes)

    # Page 4: Corrupted font ToUnicode CMap (emits \ufffd replacement chars) + header logo
    p4 = doc.new_page(width=595, height=842)
    p4.insert_image(pymupdf.Rect(420, 20, 540, 120), stream=logo_bytes)
    p4.insert_text(
        (50, 120),
        "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB texto corrompido",
        fontname="cour",
        fontsize=11,
    )
    p4_fonts = p4.get_fonts()
    cour_xref = next(f[0] for f in p4_fonts if "Courier" in str(f[3]))
    cmap = (
        b"/CIDInit /ProcSet findresource begin\n"
        b"12 dict begin\n"
        b"begincmap\n"
        b"/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        b"/CMapName /Adobe-Identity-UCS def\n"
        b"/CMapType 2 def\n"
        b"1 begincodespacerange\n<00> <FF>\nendcodespacerange\n"
        b"1 beginbfchar\n<42> <FFFD>\nendbfchar\n"
        b"endcmap\n"
        b"CMapName currentdict /CMap defineresource pop\n"
        b"end\nend"
    )
    cmap_xref = doc.get_new_xref()
    doc.update_object(cmap_xref, "<<>>")
    doc.update_stream(cmap_xref, cmap)
    doc.xref_set_key(cour_xref, "ToUnicode", f"{cmap_xref} 0 R")

    # Page 5: Vector chart (drawings + minimal text) -> visual_heavy + vector figure extraction
    p5 = doc.new_page(width=595, height=842)
    shape = p5.new_shape()
    shape.draw_rect(pymupdf.Rect(80, 140, 500, 480))
    shape.draw_line((110, 440), (470, 440))
    shape.draw_line((110, 170), (110, 440))
    shape.draw_rect(pymupdf.Rect(140, 300, 190, 440))
    shape.draw_rect(pymupdf.Rect(220, 230, 270, 440))
    shape.draw_rect(pymupdf.Rect(300, 190, 350, 440))
    shape.draw_rect(pymupdf.Rect(380, 260, 430, 440))
    shape.finish(color=(0.1, 0.2, 0.7), fill=(0.3, 0.5, 0.9), width=2)
    shape.commit()
    p5.insert_text((190, 120), "Figura Vetorial 5", fontsize=12)

    # Page 6: Clean digital text + header logo (4th occurrence -> ensures logo threshold >= 3)
    p6 = doc.new_page(width=595, height=842)
    p6.insert_image(pymupdf.Rect(420, 20, 540, 120), stream=logo_bytes)
    p6.insert_text((50, 80), "3. Conclusão e Próximos Passos", fontsize=18)
    p6.insert_text(
        (50, 130),
        "As metas de eficiência para o próximo ciclo foram integralmente validadas.",
        fontsize=11,
    )

    doc.set_toc(
        [
            [1, "1. Sumário Executivo", 1],
            [1, "2. Análise Financeira", 2],
            [1, "3. Conclusão", 6],
        ]
    )
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture()
def sample_docx_path(tmp_path: Path) -> Path:
    path = tmp_path / "contrato.docx"
    d = docx.Document()
    d.add_heading("Contrato de Prestação de Serviços", level=1)
    d.add_paragraph("Cláusula primeira: o objeto deste contrato é a extração estruturada.")
    table = d.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Item"
    table.cell(0, 1).text = "Valor"
    table.cell(1, 0).text = "Licença"
    table.cell(1, 1).text = "R$ 0,00"
    d.save(path)
    return path


@pytest.fixture()
def sample_xlsx_path(tmp_path: Path) -> Path:
    path = tmp_path / "financeiro.xlsx"
    wb = openpyxl.Workbook()
    ws1 = wb.active
    ws1.title = "Resumo"
    ws1.append(["Categoria", "Total"])
    ws1.append(["Receita", 150000])
    ws1.append(["Despesa", 90000])

    ws2 = wb.create_sheet("Detalhes")
    ws2.append(["Mes", "Valor"])
    ws2.append(["Janeiro", 50000])
    wb.save(path)
    wb.close()
    return path
