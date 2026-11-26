"""Adaptive hybrid DocumentRouter orchestrating triage, cache, engines, and visual pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable, Literal

import pymupdf

if TYPE_CHECKING:
    from mcp_gaviao_real.engines.docling_engine import DoclingEngine

from mcp_gaviao_real.config import GaviaoSettings
from mcp_gaviao_real.engines.base import ExtractedPage
from mcp_gaviao_real.engines.markitdown_engine import MarkItDownEngine
from mcp_gaviao_real.engines.pymupdf_engine import PyMuPdfEngine
from mcp_gaviao_real.hardware import detect_hardware, resolve_policy
from mcp_gaviao_real.models import (
    DocumentInspection,
    ImageRef,
    PageClass,
    PageProfile,
    ReadDocumentResult,
    SearchHit,
    SearchResponse,
    TocEntry,
)
from mcp_gaviao_real.storage.cache import DocumentCache
from mcp_gaviao_real.triage import inspect_office, inspect_pdf
from mcp_gaviao_real.utils.page_spec import (
    estimate_tokens,
    parse_page_spec,
    truncate_pages_by_budget,
)
from mcp_gaviao_real.utils.stdio_guard import redirect_stdout_to_stderr
from mcp_gaviao_real.vision.images import (
    detect_repeated_logo_hashes,
    extract_page_figures,
    render_page_image,
)
from mcp_gaviao_real.vision.vlm import VlmCaptioner

ProgressCallback = Callable[[float, float, str], Awaitable[None]]


def _breadcrumbs_for_page(toc: list[TocEntry], page_num: int) -> str:
    if not toc:
        return ""
    active_by_level: dict[int, str] = {}
    for entry in toc:
        if entry.page <= page_num:
            active_by_level[entry.level] = entry.title
            for lvl in list(active_by_level):
                if lvl > entry.level:
                    del active_by_level[lvl]
        else:
            break
    if not active_by_level:
        return ""
    return " > ".join(active_by_level[lvl] for lvl in sorted(active_by_level))


class DocumentRouter:
    """Core orchestrator for document inspection, reading, image retrieval, and search."""

    def __init__(self, settings: GaviaoSettings | None = None) -> None:
        self.settings = settings or GaviaoSettings()
        self.cache = DocumentCache(
            self.settings.cache_dir,
            max_cache_bytes=self.settings.max_cache_bytes,
        )
        self.hardware_profile = detect_hardware(self.settings)
        self.policy = resolve_policy(self.hardware_profile, self.settings)
        self.pymupdf_engine = PyMuPdfEngine()
        self.markitdown_engine = MarkItDownEngine(self.settings)
        self.vlm_captioner = VlmCaptioner(self.settings)
        self._docling_engine: DoclingEngine | None = None

    def _get_docling_engine(self) -> DoclingEngine:
        if self._docling_engine is None:
            from mcp_gaviao_real.engines.docling_engine import DoclingEngine

            self._docling_engine = DoclingEngine(self.settings, self.policy)
        return self._docling_engine

    def close(self) -> None:
        if self._docling_engine is not None:
            self._docling_engine.unload()
        self.cache.close()

    def _recommended_strategy(self, fmt: str, pages: list[PageProfile]) -> str:
        if fmt != "pdf":
            return f"Leitura rápida via MarkItDown ({self.policy.tier.value})."
        scanned = sum(
            1 for p in pages if p.classification in {PageClass.SCANNED, PageClass.CORRUPTED_FONT}
        )
        complex_p = sum(1 for p in pages if p.classification == PageClass.COMPLEX_LAYOUT)
        visual_p = sum(1 for p in pages if p.classification == PageClass.VISUAL_HEAVY)
        parts = [f"Perfil ativo: {self.policy.tier.value}."]
        if scanned:
            if self.policy.can_use_docling:
                parts.append(
                    f"{scanned} pág(s) escaneada(s)/fonte corrompida serão roteadas para Docling OCR."
                )
            else:
                parts.append(
                    f"{scanned} pág(s) escaneada(s)/fonte corrompida: use get_document_image(target='page:N') ou instale extra 'accurate'."
                )
        if complex_p:
            parts.append(f"{complex_p} pág(s) com layout complexo/tabelas.")
        if visual_p:
            parts.append(f"{visual_p} pág(s) visuais (use page_render/crop_bbox se necessário).")
        if not (scanned or complex_p or visual_p):
            parts.append("PDF digital limpo: extração instantânea via PyMuPDF.")
        return " ".join(parts)

    def inspect(self, source: str) -> DocumentInspection:
        """Inspect structure, TOC, page complexity, and image inventory (<150ms for PDFs)."""
        doc_id, path = self.cache.resolve_source(source, self.settings)
        stat = path.stat()
        ext = path.suffix.lower().lstrip(".") or "txt"

        cached_insp = self.cache.get_inspection(doc_id)
        if cached_insp is not None and cached_insp.file_size_bytes == stat.st_size:
            self.cache.touch_document(doc_id)
            return cached_insp

        images: list[ImageRef] = []
        if ext == "pdf":
            toc, pages, images = inspect_pdf(path, doc_id)
        else:
            toc, pages = inspect_office(path)

        total_tokens = sum(p.estimated_tokens for p in pages)
        inspection = DocumentInspection(
            doc_id=doc_id,
            path=str(path),
            format=ext,
            file_size_bytes=stat.st_size,
            page_count=len(pages),
            estimated_total_tokens=total_tokens,
            toc=toc,
            pages=pages,
            images=images,
            active_tier=self.policy.tier,
            recommended_strategy=self._recommended_strategy(ext, pages),
        )
        self.cache.save_document_meta(
            doc_id=doc_id,
            path=path,
            mtime_ns=stat.st_mtime_ns,
            file_size=stat.st_size,
            fmt=ext,
            page_count=len(pages),
            toc_json=json.dumps([t.model_dump() for t in toc]),
            inspection=inspection,
        )
        return inspection

    def _choose_pdf_engine(
        self,
        doc_id: str,
        page_profile: PageProfile,
        requested_mode: str,
    ) -> tuple[str, list[str]]:
        """Choose engine name and return (engine_name, warnings)."""
        warnings: list[str] = []
        p_cls = page_profile.classification
        needs_ocr = requested_mode == "ocr" or p_cls in {
            PageClass.SCANNED,
            PageClass.CORRUPTED_FONT,
        }
        needs_accurate = requested_mode == "accurate" or (
            requested_mode == "auto"
            and p_cls == PageClass.COMPLEX_LAYOUT
            and self.policy.tier.value in {"gpu_high", "gpu_low"}
        )

        if requested_mode == "fast":
            return "pymupdf_fast", warnings

        if needs_ocr or needs_accurate:
            if self.policy.can_use_docling:
                return (
                    "docling_ocr" if needs_ocr else "docling_layout",
                    warnings,
                )
            reason = (
                "escaneada/fonte corrompida"
                if p_cls in {PageClass.SCANNED, PageClass.CORRUPTED_FONT}
                else f"modo '{requested_mode}'"
            )
            warnings.append(
                f"Página {page_profile.page} ({reason}): operando em perfil '{self.policy.tier.value}' "
                f"(engine_fallback=lite_visual). Use get_document_image(source='{doc_id}', target='page:{page_profile.page}') "
                f"ou instale `{self.policy.recommended_install_cmd}`."
            )
            return "pymupdf_fast", warnings

        return "pymupdf_fast", warnings

    async def read(
        self,
        source: str,
        pages: str = "1-5",
        mode: Literal["auto", "fast", "accurate", "ocr"] = "auto",
        include_images: bool = False,
        vlm_captions: bool = False,
        max_tokens: int | None = None,
        progress_cb: ProgressCallback | None = None,
    ) -> ReadDocumentResult:
        """Read requested pages with adaptive per-page routing, cache, and token budget."""
        budget = max_tokens or self.settings.default_max_tokens
        inspection = self.inspect(source)
        doc_id = inspection.doc_id
        path = Path(inspection.path)
        page_nums = parse_page_spec(pages, inspection.page_count)

        if progress_cb:
            await progress_cb(0.1, 1.0, f"Inspeção concluída ({len(page_nums)} páginas)")

        profile_by_page = {p.page: p for p in inspection.pages}
        preferred_engines: dict[int, list[str]] = {}
        target_engine_by_page: dict[int, str] = {}
        warnings: list[str] = []

        for p_num in page_nums:
            if inspection.format == "pdf":
                eng, w_list = self._choose_pdf_engine(doc_id, profile_by_page[p_num], mode)
                warnings.extend(w_list)
                target_engine_by_page[p_num] = eng
                preferred_engines[p_num] = [eng, "pymupdf_fast"] if mode == "auto" else [eng]
            else:
                target_engine_by_page[p_num] = "markitdown_fast"
                preferred_engines[p_num] = ["markitdown_fast"]

        cached_map = self.cache.get_cached_pages(
            doc_id, page_nums, preferred_engines=preferred_engines
        )
        missing_pages = [p for p in page_nums if p not in cached_map]

        extracted_map: dict[int, ExtractedPage] = {}
        for p_num, (eng, md, _, bc) in cached_map.items():
            extracted_map[p_num] = ExtractedPage(
                page_num=p_num,
                engine=eng,
                markdown=md,
                breadcrumbs=bc,
            )

        if missing_pages:
            if inspection.format == "pdf":
                # Group missing pages by selected engine
                by_eng: dict[str, list[int]] = {}
                for p_num in missing_pages:
                    eng = target_engine_by_page[p_num]
                    by_eng.setdefault(eng, []).append(p_num)

                for eng, grp_pages in by_eng.items():
                    if eng in {"docling_layout", "docling_ocr"} and self.policy.can_use_docling:
                        do_ocr = eng == "docling_ocr"
                        run_pages = grp_pages
                        overflow_pages: list[int] = []
                        max_ocr = self.policy.max_ocr_pages_per_call
                        if do_ocr and max_ocr > 0 and len(grp_pages) > max_ocr:
                            run_pages = grp_pages[:max_ocr]
                            overflow_pages = grp_pages[max_ocr:]
                            warnings.append(
                                f"Limite max_ocr_pages_per_call={max_ocr} do perfil '{self.policy.tier.value}' atingido: "
                                f"páginas {overflow_pages} lidas via PyMuPDF rápido."
                            )
                        if progress_cb:
                            await progress_cb(
                                0.35,
                                1.0,
                                f"Extraindo páginas {run_pages} via {eng} ({self.policy.device})",
                            )
                        try:
                            extracted_list = self._get_docling_engine().extract_pages(
                                path, run_pages, doc_id=doc_id, do_ocr=do_ocr
                            )
                        except Exception as exc:
                            warnings.append(
                                f"Falha no Docling ({exc}) para páginas {run_pages}; usando fallback PyMuPDF."
                            )
                            extracted_list = self.pymupdf_engine.extract_pages(
                                path, run_pages, doc_id=doc_id
                            )
                        if overflow_pages:
                            extracted_list.extend(
                                self.pymupdf_engine.extract_pages(
                                    path, overflow_pages, doc_id=doc_id
                                )
                            )
                    else:
                        if progress_cb:
                            await progress_cb(0.35, 1.0, f"Extraindo páginas {grp_pages} via {eng}")
                        extracted_list = self.pymupdf_engine.extract_pages(
                            path, grp_pages, doc_id=doc_id
                        )
                    for item in extracted_list:
                        item.breadcrumbs = _breadcrumbs_for_page(inspection.toc, item.page_num)
                        extracted_map[item.page_num] = item
                        warnings.extend(item.warnings)
            else:
                use_docling_office = (
                    mode == "accurate"
                    and self.policy.can_use_docling
                    and inspection.format in {"docx", "pptx", "xlsx", "html", "htm"}
                )
                if use_docling_office:
                    if progress_cb:
                        await progress_cb(
                            0.35,
                            1.0,
                            f"Extraindo {inspection.format.upper()} via Docling ({self.policy.device})",
                        )
                    try:
                        extracted_list = self._get_docling_engine().extract_pages(
                            path, missing_pages, doc_id=doc_id, do_ocr=False
                        )
                    except Exception as exc:
                        warnings.append(
                            f"Falha no Docling Office ({exc}); usando fallback MarkItDown."
                        )
                        extracted_list = self.markitdown_engine.extract_pages(
                            path, missing_pages, doc_id=doc_id
                        )
                else:
                    if progress_cb:
                        await progress_cb(
                            0.35, 1.0, f"Extraindo {inspection.format.upper()} via MarkItDown"
                        )
                    extracted_list = self.markitdown_engine.extract_pages(
                        path, missing_pages, doc_id=doc_id
                    )
                for item in extracted_list:
                    if not item.breadcrumbs:
                        item.breadcrumbs = _breadcrumbs_for_page(inspection.toc, item.page_num)
                    extracted_map[item.page_num] = item
                    warnings.extend(item.warnings)

        # Visual pipeline for PDFs (anti-logo + vector figure crops)
        page_images_map: dict[int, list[ImageRef]] = {}
        if inspection.format == "pdf":
            cached_imgs = self.cache.get_page_images(doc_id, page_nums)
            for img in cached_imgs:
                page_images_map.setdefault(img.page, []).append(img)

            pages_needing_fig_scan = [
                p
                for p in page_nums
                if p not in page_images_map
                and (
                    profile_by_page[p].image_count > 0
                    or profile_by_page[p].classification
                    in {PageClass.VISUAL_HEAVY, PageClass.COMPLEX_LAYOUT}
                )
            ]
            if pages_needing_fig_scan:
                with redirect_stdout_to_stderr():
                    doc = pymupdf.open(path)
                try:
                    repeated_hashes, repeated_xrefs = detect_repeated_logo_hashes(doc)
                    for p_num in pages_needing_fig_scan:
                        if p_num in page_images_map:
                            continue
                        figs = extract_page_figures(
                            doc,
                            p_num,
                            doc_id=doc_id,
                            repeated_hashes=repeated_hashes,
                            repeated_xrefs=repeated_xrefs,
                            max_size=self.settings.default_image_max_size,
                        )
                        for ref, webp_bytes, phash in figs:
                            self.cache.save_image_blob(doc_id, ref, webp_bytes, phash=phash)
                            page_images_map.setdefault(p_num, []).append(ref)
                finally:
                    doc.close()

        # Optional VLM captions for extracted figures on requested pages
        if vlm_captions and page_images_map:
            for p_num in page_nums:
                for img in page_images_map.get(p_num, []):
                    if img.vlm_description:
                        continue
                    loaded = self.cache.get_image(doc_id, img.image_id)
                    if loaded is None:
                        continue
                    _, blob = loaded
                    desc, warn = await self.vlm_captioner.describe_image(
                        blob,
                        context_hint=extracted_map[p_num].breadcrumbs
                        if p_num in extracted_map
                        else "",
                    )
                    if warn and warn not in warnings:
                        warnings.append(warn)
                    if desc:
                        img.vlm_description = desc
                        self.cache.update_image_vlm_desc(doc_id, img.image_id, desc)

        # Persist newly extracted pages into SQLite + FTS5
        to_persist: list[tuple[int, str, str, int, str]] = []
        for p_num in missing_pages:
            item = extracted_map.get(p_num)
            if item is not None:
                to_persist.append(
                    (
                        item.page_num,
                        item.engine,
                        item.markdown,
                        estimate_tokens(item.markdown),
                        item.breadcrumbs,
                    )
                )
        if to_persist:
            self.cache.save_pages(doc_id, to_persist)
            self.cache.enforce_lru_budget()

        # Assemble per-page markdown with breadcrumbs and image placeholders
        assembled_pages: list[tuple[int, str]] = []
        for p_num in page_nums:
            item = extracted_map.get(p_num)
            if item is None:
                continue
            p_prof = profile_by_page.get(p_num)
            parts: list[str] = [f"<!-- page:{p_num} engine:{item.engine} -->"]
            if item.breadcrumbs:
                parts.append(f"<!-- seção: {item.breadcrumbs} -->")
            if item.markdown:
                parts.append(item.markdown)

            if (
                p_prof
                and p_prof.classification in {PageClass.SCANNED, PageClass.CORRUPTED_FONT}
                and item.engine == "pymupdf_fast"
            ):
                parts.append(
                    f"![Página {p_num} (renderização visual recomendada)](doc://{doc_id}/page/{p_num})"
                )

            for img in page_images_map.get(p_num, []):
                parts.append(
                    f"![{img.image_id} | {img.kind} | {img.width}x{img.height}]({img.uri})"
                )
                if img.vlm_description:
                    parts.append(f"> **Descrição VLM ({img.image_id}):** {img.vlm_description}")

            assembled_pages.append((p_num, "\n\n".join(parts)))

        kept_pages, truncated, next_page = truncate_pages_by_budget(assembled_pages, budget)
        returned_nums = [p for p, _ in kept_pages]
        engines_used: dict[str, list[int]] = {}
        returned_images: list[ImageRef] = []

        for p_num in returned_nums:
            eng = extracted_map[p_num].engine
            engines_used.setdefault(eng, []).append(p_num)
            if include_images:
                returned_images.extend(page_images_map.get(p_num, []))

        full_markdown = "\n\n---\n\n".join(text for _, text in kept_pages)

        if truncated and next_page is not None:
            warnings.append(
                f"Resposta limitada por max_tokens={budget}. Continue com pages='{next_page}-...'."
            )

        # Deduplicate warnings while preserving order
        unique_warnings = list(dict.fromkeys(warnings))
        if progress_cb:
            await progress_cb(1.0, 1.0, "Leitura concluída")

        return ReadDocumentResult(
            doc_id=doc_id,
            path=str(path),
            pages_returned=returned_nums,
            engines_used=engines_used,
            markdown=full_markdown,
            images=returned_images,
            estimated_tokens=estimate_tokens(full_markdown),
            truncated=truncated,
            next_page=next_page,
            warnings=unique_warnings,
        )

    def get_image(
        self,
        source: str,
        target: str,
        max_size: int = 1280,
        crop_bbox: list[int] | None = None,
    ) -> tuple[dict[str, object], bytes]:
        """Retrieve an extracted figure ('p3_fig1') or rendered/zoomed PDF page ('page:3')."""
        inspection = self.inspect(source)
        doc_id = inspection.doc_id
        path = Path(inspection.path)
        t = target.strip()

        page_match = re.fullmatch(r"page:(\d+)", t, flags=re.IGNORECASE)
        if page_match:
            if inspection.format != "pdf":
                raise ValueError(
                    "Renderização de página ('page:N') só é suportada para arquivos PDF."
                )
            page_num = int(page_match.group(1))
            webp_bytes, w, h, kind = render_page_image(
                path,
                page_num,
                max_size=max_size,
                crop_bbox=crop_bbox,
            )
            meta: dict[str, object] = {
                "doc_id": doc_id,
                "target": t,
                "kind": kind,
                "page": page_num,
                "width": w,
                "height": h,
                "mime_type": "image/webp",
                "crop_bbox": crop_bbox,
            }
            return meta, webp_bytes

        if crop_bbox is not None:
            raise ValueError(
                "crop_bbox só pode ser usado em alvos do tipo 'page:N' (ex.: target='page:3')."
            )

        loaded = self.cache.get_image(doc_id, t)
        if loaded is None and inspection.format == "pdf":
            # Populate figures on demand if user passed image_id before calling read_document
            m_fig = re.match(r"p(\d+)_fig\d+", t)
            if m_fig:
                p_num = int(m_fig.group(1))
                if 1 <= p_num <= inspection.page_count:
                    with redirect_stdout_to_stderr():
                        doc = pymupdf.open(path)
                    try:
                        repeated_hashes, repeated_xrefs = detect_repeated_logo_hashes(doc)
                        figs = extract_page_figures(
                            doc,
                            p_num,
                            doc_id=doc_id,
                            repeated_hashes=repeated_hashes,
                            repeated_xrefs=repeated_xrefs,
                            max_size=max_size,
                        )
                        for ref, webp_bytes, phash in figs:
                            self.cache.save_image_blob(doc_id, ref, webp_bytes, phash=phash)
                    finally:
                        doc.close()
                    loaded = self.cache.get_image(doc_id, t)

        if loaded is None:
            raise FileNotFoundError(
                f"Imagem '{t}' não encontrada no documento '{doc_id}'. "
                f"Use read_document ou inspect_document para listar IDs válidos, ou target='page:N' para renderizar a página."
            )

        ref, blob = loaded
        if max_size and max(ref.width, ref.height) > max_size:
            import io

            from PIL import Image

            from mcp_gaviao_real.vision.images import encode_webp

            with Image.open(io.BytesIO(blob)) as im:
                blob, new_w, new_h = encode_webp(im, max_size=max_size)
        else:
            new_w, new_h = ref.width, ref.height

        meta = {
            "doc_id": doc_id,
            "target": t,
            "kind": ref.kind,
            "page": ref.page,
            "width": new_w,
            "height": new_h,
            "bbox": ref.bbox,
            "mime_type": "image/webp",
            "vlm_description": ref.vlm_description,
        }
        return meta, blob

    def search(
        self,
        source: str,
        query: str,
        mode: Literal["fts", "regex", "hybrid"] = "hybrid",
        limit: int = 10,
    ) -> SearchResponse:
        """Search document using SQLite FTS5 (accent-insensitive BM25) and/or regex."""
        if not query or not query.strip():
            raise ValueError("A consulta de busca (query) não pode estar vazia.")

        inspection = self.inspect(source)
        doc_id = inspection.doc_id
        path = Path(inspection.path)

        indexed_pages = self.cache.get_indexed_page_nums(doc_id)
        unindexed_scanned: list[int] = []
        pages_to_index: list[int] = []

        for p_prof in inspection.pages:
            if p_prof.classification in {PageClass.SCANNED, PageClass.CORRUPTED_FONT}:
                if p_prof.page not in indexed_pages:
                    unindexed_scanned.append(p_prof.page)
                continue
            if p_prof.page not in indexed_pages:
                pages_to_index.append(p_prof.page)

        if pages_to_index:
            if inspection.format == "pdf":
                extracted = self.pymupdf_engine.extract_pages(path, pages_to_index, doc_id=doc_id)
            else:
                extracted = self.markitdown_engine.extract_pages(
                    path, pages_to_index, doc_id=doc_id
                )
            persist_rows = [
                (
                    item.page_num,
                    item.engine,
                    item.markdown,
                    estimate_tokens(item.markdown),
                    item.breadcrumbs or _breadcrumbs_for_page(inspection.toc, item.page_num),
                )
                for item in extracted
            ]
            self.cache.save_pages(doc_id, persist_rows)

        hits_by_page: dict[int, SearchHit] = {}
        if mode in {"fts", "hybrid"}:
            for hit in self.cache.search_fts(doc_id, query, limit=limit):
                hits_by_page[hit.page] = hit

        if mode in {"regex", "hybrid"}:
            try:
                regex_hits = self.cache.search_regex(doc_id, query, limit=limit)
            except re.error:
                if mode == "regex":
                    raise
                regex_hits = []
            for hit in regex_hits:
                if hit.page not in hits_by_page:
                    hits_by_page[hit.page] = hit

        ordered_hits = list(hits_by_page.values())[:limit]
        return SearchResponse(
            doc_id=doc_id,
            query=query,
            mode=mode,
            total_hits=len(ordered_hits),
            hits=ordered_hits,
            unindexed_scanned_pages=sorted(unindexed_scanned),
        )
