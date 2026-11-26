"""Visual pipeline: anti-noise/anti-logo filtering, vector chart crop, WebP encoding, and 300 DPI zoom."""

from __future__ import annotations

import io
from collections import Counter
from pathlib import Path

import pymupdf
from PIL import Image, ImageStat

from mcp_gaviao_real.models import ImageRef
from mcp_gaviao_real.utils.stdio_guard import redirect_stdout_to_stderr


def compute_dhash(img: Image.Image, hash_size: int = 8) -> str:
    """Compute a 64-bit difference hash (dHash) using pure Pillow."""
    gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    if hasattr(gray, "get_flattened_data"):
        pixels = list(gray.get_flattened_data())
    else:
        pixels = list(gray.getdata())
    bits = 0
    for row in range(hash_size):
        row_offset = row * (hash_size + 1)
        for col in range(hash_size):
            bits = (bits << 1) | int(pixels[row_offset + col] > pixels[row_offset + col + 1])
    return f"{bits:016x}"


def is_noise_image(img: Image.Image) -> bool:
    """Filter tiny icons, extreme banner lines, and near-monochromatic blocks."""
    w, h = img.size
    if w < 80 or h < 80:
        return True
    aspect = max(w / max(1, h), h / max(1, w))
    if aspect > 15.0:
        return True
    stat = ImageStat.Stat(img.convert("L"))
    return bool(stat.stddev and stat.stddev[0] < 5.0)


def encode_webp(
    img: Image.Image,
    max_size: int = 1280,
    quality: int = 84,
) -> tuple[bytes, int, int]:
    """Resize preserving aspect ratio so max side <= max_size and encode as WebP."""
    work = img.convert("RGB")
    w, h = work.size
    max_dim = max(w, h)
    if max_size > 0 and max_dim > max_size:
        scale = max_size / float(max_dim)
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        work = work.resize((new_w, new_h), Image.Resampling.LANCZOS)
        w, h = work.size

    buf = io.BytesIO()
    work.save(buf, format="WEBP", quality=quality, method=4)
    return buf.getvalue(), w, h


def _pixmap_to_pil(pix: pymupdf.Pixmap) -> Image.Image:
    if pix.alpha:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def detect_repeated_logo_hashes(
    doc: pymupdf.Document,
    threshold: int = 3,
    max_scan_pages: int = 30,
) -> tuple[set[str], set[int]]:
    """Identify recurring header/footer logos (dHashes and xrefs) across >= threshold pages."""
    hash_pages: dict[str, set[int]] = {}
    xref_pages: Counter[int] = Counter()

    scan_limit = min(len(doc), max_scan_pages)
    for p_idx in range(scan_limit):
        page = doc[p_idx]
        seen_xrefs_on_page: set[int] = set()
        for info in page.get_image_info(xrefs=True) or []:
            xref = int(info.get("xref", 0) or 0)
            w = int(info.get("width", 0) or 0)
            h = int(info.get("height", 0) or 0)
            if xref <= 0 or w < 40 or h < 40:
                continue
            if xref not in seen_xrefs_on_page:
                seen_xrefs_on_page.add(xref)
                xref_pages[xref] += 1

    repeated_xrefs = {xref for xref, count in xref_pages.items() if count >= threshold}
    repeated_hashes: set[str] = set()

    for p_idx in range(scan_limit):
        page = doc[p_idx]
        for info in page.get_image_info(xrefs=True) or []:
            bbox = info.get("bbox")
            xref = int(info.get("xref", 0) or 0)
            w = int(info.get("width", 0) or 0)
            h = int(info.get("height", 0) or 0)
            if not bbox or w < 60 or h < 60:
                continue
            rect = pymupdf.Rect(bbox) & page.rect
            if rect.width < 30 or rect.height < 30:
                continue
            if (rect.width * rect.height) > (page.rect.width * page.rect.height * 0.25):
                continue
            try:
                pix = page.get_pixmap(clip=rect, dpi=180, alpha=False)
                pil_img = _pixmap_to_pil(pix)
                dh = compute_dhash(pil_img)
                if xref in repeated_xrefs:
                    repeated_hashes.add(dh)
                else:
                    hash_pages.setdefault(dh, set()).add(p_idx)
            except Exception:
                continue

    for dh, pages_set in hash_pages.items():
        if len(pages_set) >= threshold:
            repeated_hashes.add(dh)

    return repeated_hashes, repeated_xrefs


def _rect_intersects_any(rect: pymupdf.Rect, existing: list[pymupdf.Rect]) -> bool:
    for other in existing:
        inter = rect & other
        if not inter.is_empty:
            inter_area = inter.width * inter.height
            min_area = max(1.0, min(rect.width * rect.height, other.width * other.height))
            if inter_area / min_area > 0.5:
                return True
    return False


def extract_page_figures(
    doc: pymupdf.Document,
    page_num: int,
    doc_id: str,
    repeated_hashes: set[str],
    repeated_xrefs: set[int] | None = None,
    max_size: int = 1280,
) -> list[tuple[ImageRef, bytes, str]]:
    """Extract meaningful embedded images and vector graphic regions from a PDF page."""
    page = doc[page_num - 1]
    candidate_rects: list[tuple[str, pymupdf.Rect]] = []
    seen_rects: list[pymupdf.Rect] = []
    skip_xrefs = repeated_xrefs or set()

    for info in page.get_image_info(xrefs=True) or []:
        bbox = info.get("bbox")
        xref = int(info.get("xref", 0) or 0)
        w = int(info.get("width", 0) or 0)
        h = int(info.get("height", 0) or 0)
        if not bbox or w < 80 or h < 80:
            continue
        if xref in skip_xrefs:
            continue
        rect = pymupdf.Rect(bbox) & page.rect
        if rect.width < 40 or rect.height < 40:
            continue
        if not _rect_intersects_any(rect, seen_rects):
            seen_rects.append(rect)
            candidate_rects.append(("figure", rect))

    # Capture vector charts / diagrams (drawings + nearby labels)
    try:
        clusters = page.cluster_drawings() or []
    except Exception:
        clusters = []

    for c_rect in clusters:
        rect = pymupdf.Rect(c_rect) & page.rect
        if rect.width < 100 or rect.height < 80:
            continue
        # Expand slightly to include axis tick labels around vector plots
        padded = pymupdf.Rect(rect.x0 - 14, rect.y0 - 14, rect.x1 + 14, rect.y1 + 14) & page.rect
        if not _rect_intersects_any(padded, seen_rects):
            seen_rects.append(padded)
            candidate_rects.append(("vector_chart", padded))

    extracted: list[tuple[ImageRef, bytes, str]] = []
    fig_counter = 0

    for kind, rect in candidate_rects:
        pix = page.get_pixmap(clip=rect, dpi=180, alpha=False)
        pil_img = _pixmap_to_pil(pix)
        if is_noise_image(pil_img):
            continue
        dhash = compute_dhash(pil_img)
        if dhash in repeated_hashes:
            continue

        fig_counter += 1
        image_id = f"p{page_num}_fig{fig_counter}"
        webp_bytes, out_w, out_h = encode_webp(pil_img, max_size=max_size)
        ref = ImageRef(
            image_id=image_id,
            page=page_num,
            kind=kind,
            width=out_w,
            height=out_h,
            uri=f"doc://{doc_id}/img/{image_id}",
            bbox=[
                round(float(rect.x0), 1),
                round(float(rect.y0), 1),
                round(float(rect.x1), 1),
                round(float(rect.y1), 1),
            ],
        )
        extracted.append((ref, webp_bytes, dhash))

    return extracted


def render_page_image(
    path: Path,
    page_num: int,
    max_size: int = 1280,
    crop_bbox: list[int] | None = None,
) -> tuple[bytes, int, int, str]:
    """Render a full PDF page (150 DPI) or normalized 0..1000 crop_bbox (300 DPI eagle-eye zoom)."""
    with redirect_stdout_to_stderr():
        doc = pymupdf.open(path)
    try:
        if page_num < 1 or page_num > len(doc):
            raise ValueError(f"Página {page_num} fora dos limites do documento (1..{len(doc)}).")
        page = doc[page_num - 1]
        rect = page.rect

        if crop_bbox is not None:
            if len(crop_bbox) != 4:
                raise ValueError(
                    "crop_bbox deve conter exatamente 4 inteiros [x0, y0, x1, y1] na escala 0..1000."
                )
            x0, y0, x1, y1 = crop_bbox
            if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
                raise ValueError(
                    f"crop_bbox inválido {crop_bbox}: requer 0 <= x0 < x1 <= 1000 e 0 <= y0 < y1 <= 1000."
                )
            clip = pymupdf.Rect(
                rect.x0 + (x0 / 1000.0) * rect.width,
                rect.y0 + (y0 / 1000.0) * rect.height,
                rect.x0 + (x1 / 1000.0) * rect.width,
                rect.y0 + (y1 / 1000.0) * rect.height,
            )
            pix = page.get_pixmap(clip=clip, dpi=300, alpha=False)
            kind = "zoom_crop"
        else:
            pix = page.get_pixmap(dpi=150, alpha=False)
            kind = "page_render"

        pil_img = _pixmap_to_pil(pix)
        webp_bytes, out_w, out_h = encode_webp(pil_img, max_size=max_size)
        return webp_bytes, out_w, out_h, kind
    finally:
        doc.close()
