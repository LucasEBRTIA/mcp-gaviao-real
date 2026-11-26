"""Tests for SQLite FTS5 accent-insensitive search, regex, breadcrumbs, and scanned page warnings."""

from __future__ import annotations

from pathlib import Path

from mcp_gaviao_real.router import DocumentRouter


def test_fts5_accent_insensitive_and_unindexed_scanned_pages(
    router: DocumentRouter,
    mixed_pdf_path: Path,
) -> None:
    # Query without Portuguese accents must match "Relatório Líquido" on page 1
    resp = router.search(str(mixed_pdf_path), "relatorio liquido", mode="fts")
    assert resp.total_hits >= 1
    assert any(hit.page == 1 for hit in resp.hits)
    assert any("Sumário Executivo" in hit.breadcrumbs for hit in resp.hits)
    # Pages 3 (scanned) and 4 (corrupted font) must be flagged as unindexed
    assert 3 in resp.unindexed_scanned_pages
    assert 4 in resp.unindexed_scanned_pages


def test_regex_and_hybrid_search(
    router: DocumentRouter,
    mixed_pdf_path: Path,
) -> None:
    resp = router.search(str(mixed_pdf_path), r"18,5%", mode="regex")
    assert resp.total_hits == 1
    assert resp.hits[0].page == 1

    hybrid = router.search(str(mixed_pdf_path), "trimestral", mode="hybrid")
    assert any(hit.page == 2 for hit in hybrid.hits)
