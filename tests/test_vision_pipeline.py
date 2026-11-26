"""Tests for visual pipeline: anti-logo filtering, vector chart extraction, WebP, and 300 DPI zoom."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from mcp_gaviao_real.router import DocumentRouter


@pytest.mark.asyncio
async def test_anti_logo_suppression_and_chart_extraction(
    router: DocumentRouter,
    mixed_pdf_path: Path,
) -> None:
    # Page 1 has only the repeated header logo (present on pages 1,2,4,6) -> must be suppressed
    res_p1 = await router.read(str(mixed_pdf_path), pages="1", include_images=True)
    assert len(res_p1.images) == 0

    # Page 2 has the repeated header logo + a distinct bar chart -> only the chart must remain
    res_p2 = await router.read(str(mixed_pdf_path), pages="2", include_images=True)
    assert len(res_p2.images) == 1
    chart_ref = res_p2.images[0]
    assert chart_ref.image_id == "p2_fig1"
    assert chart_ref.uri in res_p2.markdown

    # Page 5 has a vector chart drawn via PDF paths -> must be captured as vector_chart
    res_p5 = await router.read(str(mixed_pdf_path), pages="5", include_images=True)
    assert len(res_p5.images) >= 1
    assert any(img.kind == "vector_chart" for img in res_p5.images)


@pytest.mark.asyncio
async def test_get_image_webp_and_zoom_crop(
    router: DocumentRouter,
    mixed_pdf_path: Path,
) -> None:
    res_p2 = await router.read(str(mixed_pdf_path), pages="2", include_images=True)
    fig_id = res_p2.images[0].image_id

    meta, webp_bytes = router.get_image(res_p2.doc_id, fig_id, max_size=400)
    assert meta["mime_type"] == "image/webp"
    with Image.open(io.BytesIO(webp_bytes)) as im:
        assert im.format == "WEBP"
        assert max(im.size) <= 400

    # Full page render for scanned page 3
    p_meta, p_bytes = router.get_image(res_p2.doc_id, "page:3", max_size=800)
    assert p_meta["kind"] == "page_render"
    with Image.open(io.BytesIO(p_bytes)) as im:
        assert max(im.size) <= 800

    # 300 DPI eagle-eye zoom crop on bottom-right quadrant
    z_meta, z_bytes = router.get_image(
        res_p2.doc_id,
        "page:5",
        max_size=1280,
        crop_bbox=[100, 150, 900, 600],
    )
    assert z_meta["kind"] == "zoom_crop"
    with Image.open(io.BytesIO(z_bytes)) as im:
        assert im.format == "WEBP"
        assert im.width > 400 and im.height > 400

    with pytest.raises(ValueError):
        router.get_image(res_p2.doc_id, "page:5", crop_bbox=[600, 100, 400, 500])
