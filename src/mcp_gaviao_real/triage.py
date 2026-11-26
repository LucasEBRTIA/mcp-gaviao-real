"""Fast deterministic document triage (<150ms) for PDFs and Office files."""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pymupdf

from mcp_gaviao_real.models import ImageRef, PageClass, PageProfile, TocEntry
from mcp_gaviao_real.utils.page_spec import estimate_tokens
from mcp_gaviao_real.utils.stdio_guard import redirect_stdout_to_stderr


def _has_multi_column_layout(blocks: list[tuple[object, ...]], page_width: float) -> bool:
    """Detect multi-column layout by checking side-by-side non-overlapping text blocks."""
    text_boxes: list[tuple[float, float, float, float]] = []
    for b in blocks:
        # PyMuPDF block tuple: (x0, y0, x1, y1, text, block_no, block_type)
        if len(b) >= 7 and int(b[6]) == 0 and str(b[4]).strip():
            x0, y0, x1, y1 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            if (x1 - x0) < page_width * 0.55 and (y1 - y0) > 12:
                text_boxes.append((x0, y0, x1, y1))

    if len(text_boxes) < 4:
        return False

    mid_x = page_width * 0.5
    left_blocks = [b for b in text_boxes if b[2] <= mid_x + 15]
    right_blocks = [b for b in text_boxes if b[0] >= mid_x - 15]
    if len(left_blocks) < 2 or len(right_blocks) < 2:
        return False

    # Confirm vertical overlap between left and right column blocks
    for lb in left_blocks:
        for rb in right_blocks:
            if min(lb[3], rb[3]) - max(lb[1], rb[1]) > 10:
                return True
    return False


def _compute_garbled_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad = 0
    total = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        code = ord(ch)
        if ch == "\ufffd" or code == 0 or (0xE000 <= code <= 0xF8FF):
            bad += 1
    if total == 0:
        return 0.0
    return round(bad / total, 4)


def inspect_pdf(
    path: Path,
    doc_id: str,
) -> tuple[list[TocEntry], list[PageProfile], list[ImageRef]]:
    """Inspect PDF structure, TOC, and per-page complexity without ML models."""
    with redirect_stdout_to_stderr():
        doc = pymupdf.open(path)
    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise ValueError(f"PDF protegido por senha/criptografado: '{path.name}'.")

        raw_toc = doc.get_toc(simple=True) or []
        toc = [
            TocEntry(level=int(item[0]), title=str(item[1]).strip(), page=max(1, int(item[2])))
            for item in raw_toc
            if len(item) >= 3 and str(item[1]).strip()
        ]

        pages: list[PageProfile] = []
        image_refs: list[ImageRef] = []

        for idx in range(len(doc)):
            page_num = idx + 1
            page = doc[idx]
            rect = page.rect
            page_area = max(1.0, float(rect.width * rect.height))

            text = page.get_text("text", flags=pymupdf.TEXT_MEDIABOX_CLIP) or ""
            char_count = len(text.strip())
            garbled_ratio = _compute_garbled_ratio(text)
            blocks = page.get_text("blocks") or []

            img_infos = page.get_image_info(xrefs=True) or []
            img_area = 0.0
            valid_img_count = 0
            for info_idx, info in enumerate(img_infos):
                bbox = info.get("bbox")
                w = int(info.get("width", 0))
                h = int(info.get("height", 0))
                if w < 40 or h < 40 or not bbox:
                    continue
                valid_img_count += 1
                bx0, by0, bx1, by1 = [float(v) for v in bbox]
                area = max(0.0, (bx1 - bx0) * (by1 - by0))
                img_area += area
                image_refs.append(
                    ImageRef(
                        image_id=f"p{page_num}_raw{info_idx + 1}",
                        page=page_num,
                        kind="embedded",
                        width=w,
                        height=h,
                        uri=f"doc://{doc_id}/img/p{page_num}_raw{info_idx + 1}",
                        bbox=[round(bx0, 1), round(by0, 1), round(bx1, 1), round(by1, 1)],
                    )
                )

            image_coverage = round(min(1.0, img_area / page_area), 3)
            drawings = page.get_drawings() or []
            drawing_ops = len(drawings)
            try:
                drawing_clusters = page.cluster_drawings() or []
            except Exception:
                drawing_clusters = []
            drawing_area = sum(
                max(0.0, float(pymupdf.Rect(c).width * pymupdf.Rect(c).height))
                for c in drawing_clusters
            )
            drawing_coverage = min(1.0, drawing_area / page_area)
            table_hint = drawing_ops >= 8 or "\t" in text
            multi_col = _has_multi_column_layout(blocks, float(rect.width))

            suggest_render = False
            if char_count >= 20 and garbled_ratio > 0.05:
                classification = PageClass.CORRUPTED_FONT
                suggest_render = True
            elif char_count < 30 and (image_coverage >= 0.40 or drawing_ops > 500):
                classification = PageClass.SCANNED
                suggest_render = True
            elif char_count < 120 and (
                drawing_ops >= 15 or drawing_coverage >= 0.12 or image_coverage >= 0.25
            ):
                classification = PageClass.VISUAL_HEAVY
                suggest_render = True
            elif multi_col or (table_hint and drawing_ops >= 16):
                classification = PageClass.COMPLEX_LAYOUT
            else:
                classification = PageClass.DIGITAL_CLEAN

            pages.append(
                PageProfile(
                    page=page_num,
                    classification=classification,
                    char_count=char_count,
                    image_count=valid_img_count,
                    image_coverage=image_coverage,
                    garbled_ratio=garbled_ratio,
                    table_hint=table_hint,
                    suggest_page_render=suggest_render,
                    estimated_tokens=estimate_tokens(text),
                )
            )

        return toc, pages, image_refs
    finally:
        doc.close()


def inspect_office(path: Path) -> tuple[list[TocEntry], list[PageProfile]]:
    """Fast structural inspection for Office, HTML, CSV, and text formats."""
    ext = path.suffix.lower()
    toc: list[TocEntry] = []
    pages: list[PageProfile] = []

    if ext in {".xlsx", ".xlsm"}:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            for idx, name in enumerate(wb.sheetnames, start=1):
                ws = wb[name]
                row_count = ws.max_row or 0
                col_count = ws.max_column or 0
                est_chars = min(row_count, 200) * max(1, col_count) * 14
                toc.append(TocEntry(level=1, title=name, page=idx))
                pages.append(
                    PageProfile(
                        page=idx,
                        classification=PageClass.COMPLEX_LAYOUT
                        if col_count > 6
                        else PageClass.DIGITAL_CLEAN,
                        char_count=est_chars,
                        image_count=0,
                        image_coverage=0.0,
                        garbled_ratio=0.0,
                        table_hint=True,
                        suggest_page_render=False,
                        estimated_tokens=max(10, est_chars // 4),
                    )
                )
        finally:
            wb.close()
        return toc, pages

    if ext == ".pptx":
        try:
            from pptx import Presentation

            prs = Presentation(str(path))
            for idx, slide in enumerate(prs.slides, start=1):
                texts = [
                    shape.text.strip()
                    for shape in slide.shapes
                    if getattr(shape, "has_text_frame", False) and shape.text.strip()
                ]
                combined = "\n".join(texts)
                title = texts[0].splitlines()[0][:80] if texts else f"Slide {idx}"
                toc.append(TocEntry(level=1, title=title, page=idx))
                pages.append(
                    PageProfile(
                        page=idx,
                        classification=PageClass.VISUAL_HEAVY
                        if len(combined) < 80
                        else PageClass.DIGITAL_CLEAN,
                        char_count=len(combined),
                        image_count=sum(1 for s in slide.shapes if s.shape_type == 13),
                        image_coverage=0.0,
                        garbled_ratio=0.0,
                        table_hint=any(getattr(s, "has_table", False) for s in slide.shapes),
                        suggest_page_render=False,
                        estimated_tokens=max(10, estimate_tokens(combined)),
                    )
                )
            if pages:
                return toc, pages
        except Exception:
            pass

    size_bytes = path.stat().st_size
    est_chars = min(size_bytes, 250_000)
    pages.append(
        PageProfile(
            page=1,
            classification=PageClass.DIGITAL_CLEAN,
            char_count=est_chars,
            image_count=0,
            image_coverage=0.0,
            garbled_ratio=0.0,
            table_hint=ext in {".csv", ".tsv", ".html", ".htm"},
            suggest_page_render=False,
            estimated_tokens=max(10, est_chars // 4),
        )
    )
    return toc, pages
