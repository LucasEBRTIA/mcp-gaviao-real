"""Tests for adaptive DoclingEngine, hybrid routing, and GPU VRAM lifecycle."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from mcp_gaviao_real.config import GaviaoSettings
from mcp_gaviao_real.engines.base import ExtractedPage
from mcp_gaviao_real.engines.docling_engine import DoclingEngine, group_contiguous_ranges
from mcp_gaviao_real.hardware import detect_hardware, resolve_policy
from mcp_gaviao_real.models import HardwareProfile
from mcp_gaviao_real.router import DocumentRouter


def test_group_contiguous_ranges() -> None:
    assert group_contiguous_ranges([]) == []
    assert group_contiguous_ranges([1, 2, 3, 7, 8, 10]) == [(1, 3), (7, 8), (10, 10)]
    assert group_contiguous_ranges([5, 3, 4, 3]) == [(3, 5)]


def test_docling_engine_lazy_init_and_unload(isolated_settings: GaviaoSettings) -> None:
    prof = HardwareProfile(
        os_name="Windows",
        arch="AMD64",
        cpu_cores=14,
        ram_total_gb=31.6,
        ram_available_gb=12.0,
        gpu_vendor="nvidia",
        gpu_name="RTX 4060 Ti",
        vram_total_mb=16380,
        has_docling=True,
        has_torch=True,
    )
    policy = resolve_policy(prof, isolated_settings)
    engine = DoclingEngine(isolated_settings, policy)
    assert engine.is_loaded is False
    engine.unload()
    assert engine.is_loaded is False


@pytest.mark.asyncio
async def test_router_routes_scanned_to_docling_when_enabled(
    tmp_path: Path,
    mixed_pdf_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = GaviaoSettings(cache_dir=tmp_path / "gpu_cache", profile="gpu_high")
    router = DocumentRouter(settings)
    # Force can_use_docling=True in case test runs in lite env
    router.policy.can_use_docling = True

    calls: list[tuple[list[int], bool]] = []

    class FakeDocling:
        def extract_pages(
            self, path: Path, page_nums: list[int], *, doc_id: str, do_ocr: bool = False
        ) -> list[ExtractedPage]:
            calls.append((page_nums, do_ocr))
            return [
                ExtractedPage(
                    page_num=p,
                    engine="docling_ocr" if do_ocr else "docling_layout",
                    markdown=f"OCR texto página {p}",
                )
                for p in page_nums
            ]

        def unload(self) -> None:
            pass

    monkeypatch.setattr(router, "_get_docling_engine", lambda: FakeDocling())

    try:
        res = await router.read(str(mixed_pdf_path), pages="1,3", mode="auto")
        assert res.pages_returned == [1, 3]
        assert res.engines_used.get("pymupdf_fast") == [1]
        assert res.engines_used.get("docling_ocr") == [3]
        assert calls == [([3], True)]
        assert "OCR texto página 3" in res.markdown
    finally:
        router.close()


@pytest.mark.skipif(
    importlib.util.find_spec("docling") is None,
    reason="Extra 'accurate' (docling) não instalado neste ambiente.",
)
def test_docling_real_cuda_extraction(
    isolated_settings: GaviaoSettings,
    mixed_pdf_path: Path,
) -> None:
    prof = detect_hardware(isolated_settings)
    policy = resolve_policy(prof, isolated_settings)
    assert policy.can_use_docling is True

    engine = DoclingEngine(isolated_settings, policy)
    assert engine.is_loaded is False
    try:
        pages = engine.extract_pages(
            mixed_pdf_path,
            [1],
            doc_id="test_real",
            do_ocr=False,
        )
        assert len(pages) == 1
        assert pages[0].page_num == 1
        assert "Sumário Executivo" in pages[0].markdown or "Relatório" in pages[0].markdown
        if not policy.clear_vram_after_job:
            assert engine.is_loaded is True
    finally:
        engine.unload()
        assert engine.is_loaded is False
