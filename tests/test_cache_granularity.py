"""Tests for page-granular SQLite caching, fast-hash, token budget, and LRU eviction."""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_gaviao_real.config import GaviaoSettings
from mcp_gaviao_real.models import ImageRef
from mcp_gaviao_real.router import DocumentRouter
from mcp_gaviao_real.storage.cache import DocumentCache


@pytest.mark.asyncio
async def test_only_requested_pages_are_cached(
    router: DocumentRouter,
    mixed_pdf_path: Path,
) -> None:
    res = await router.read(str(mixed_pdf_path), pages="1-2")
    assert res.pages_returned == [1, 2]

    rows = router.cache._conn.execute(
        "SELECT page_num FROM pages WHERE doc_id = ? ORDER BY page_num",
        (res.doc_id,),
    ).fetchall()
    assert [int(r["page_num"]) for r in rows] == [1, 2]


def test_fast_fingerprint_hit_without_rehash(
    isolated_settings: GaviaoSettings,
    mixed_pdf_path: Path,
) -> None:
    cache = DocumentCache(isolated_settings.cache_dir)
    try:
        doc_id1, mtime1, size1, hit1 = cache.fingerprint_file(mixed_pdf_path)
        assert hit1 is False
        cache.save_document_meta(doc_id1, mixed_pdf_path, mtime1, size1, "pdf", 6, "[]")

        doc_id2, _, _, hit2 = cache.fingerprint_file(mixed_pdf_path)
        assert hit2 is True
        assert doc_id2 == doc_id1
    finally:
        cache.close()


@pytest.mark.asyncio
async def test_max_tokens_truncation_returns_next_page(
    router: DocumentRouter,
    mixed_pdf_path: Path,
) -> None:
    # Request pages 1,2,6 with a tight token budget so it truncates cleanly
    res = await router.read(str(mixed_pdf_path), pages="1,2,6", max_tokens=130)
    assert res.truncated is True
    assert res.next_page is not None
    assert len(res.pages_returned) < 3


def test_lru_cache_eviction(tmp_path: Path) -> None:
    cache = DocumentCache(tmp_path / "lru_cache", max_cache_bytes=15_000)
    try:
        p1 = tmp_path / "doc1.txt"
        p2 = tmp_path / "doc2.txt"
        p1.write_text("alpha", encoding="utf-8")
        p2.write_text("beta", encoding="utf-8")

        cache.save_document_meta("0000000000000001", p1, 1, 5, "txt", 1, "[]")
        cache.save_image_blob(
            "0000000000000001",
            ImageRef(
                image_id="p1_fig1",
                page=1,
                kind="figure",
                width=100,
                height=100,
                uri="doc://1/img/p1_fig1",
            ),
            b"x" * 10_000,
        )
        cache.save_document_meta("0000000000000002", p2, 1, 4, "txt", 1, "[]")
        cache.save_image_blob(
            "0000000000000002",
            ImageRef(
                image_id="p1_fig1",
                page=1,
                kind="figure",
                width=100,
                height=100,
                uri="doc://2/img/p1_fig1",
            ),
            b"y" * 10_000,
        )

        evicted = cache.enforce_lru_budget()
        assert evicted >= 1
        assert cache.get_image("0000000000000001", "p1_fig1") is None
    finally:
        cache.close()
