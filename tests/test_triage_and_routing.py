"""Tests for fast PDF/Office triage and adaptive routing."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from mcp_gaviao_real.config import GaviaoSettings
from mcp_gaviao_real.models import HardwareTier, PageClass
from mcp_gaviao_real.router import DocumentRouter


def test_pdf_triage_classification_and_speed(
    router: DocumentRouter,
    mixed_pdf_path: Path,
) -> None:
    t0 = time.perf_counter()
    insp = router.inspect(str(mixed_pdf_path))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert insp.page_count == 6
    assert insp.format == "pdf"
    assert len(insp.toc) == 3
    assert elapsed_ms < 500  # generous CI floor; typically <50ms

    by_page = {p.page: p for p in insp.pages}
    assert by_page[1].classification == PageClass.DIGITAL_CLEAN
    assert by_page[2].classification == PageClass.DIGITAL_CLEAN
    assert by_page[3].classification == PageClass.SCANNED
    assert by_page[3].suggest_page_render is True
    assert by_page[4].classification == PageClass.CORRUPTED_FONT
    assert by_page[4].suggest_page_render is True
    assert by_page[5].classification == PageClass.VISUAL_HEAVY
    assert by_page[5].suggest_page_render is True
    assert by_page[6].classification == PageClass.DIGITAL_CLEAN


@pytest.mark.asyncio
async def test_lite_graceful_degradation_on_scanned_pages(
    tmp_path: Path,
    mixed_pdf_path: Path,
) -> None:
    lite_router = DocumentRouter(GaviaoSettings(cache_dir=tmp_path / "lite_cache", profile="lite"))
    try:
        assert lite_router.policy.tier == HardwareTier.LITE
        res = await lite_router.read(str(mixed_pdf_path), pages="3-4", mode="auto")

        assert res.pages_returned == [3, 4]
        assert "pymupdf_fast" in res.engines_used
        assert len(res.warnings) >= 2
        assert any("lite_visual" in w for w in res.warnings)
        assert f"doc://{res.doc_id}/page/3" in res.markdown
        assert f"doc://{res.doc_id}/page/4" in res.markdown
    finally:
        lite_router.close()


@pytest.mark.asyncio
async def test_office_docx_and_xlsx_reading(
    router: DocumentRouter,
    sample_docx_path: Path,
    sample_xlsx_path: Path,
) -> None:
    docx_res = await router.read(str(sample_docx_path), pages="1")
    assert "Cláusula primeira" in docx_res.markdown
    assert "Licença" in docx_res.markdown

    xlsx_insp = router.inspect(str(sample_xlsx_path))
    assert xlsx_insp.page_count == 2
    assert [t.title for t in xlsx_insp.toc] == ["Resumo", "Detalhes"]

    xlsx_res = await router.read(str(sample_xlsx_path), pages="2")
    assert "Janeiro" in xlsx_res.markdown
    assert "Receita" not in xlsx_res.markdown
