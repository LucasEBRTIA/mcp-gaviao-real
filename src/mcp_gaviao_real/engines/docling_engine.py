"""Adaptive lazy-loaded Docling v2 extraction engine with idle VRAM unloading."""

from __future__ import annotations

import gc
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

from mcp_gaviao_real.config import GaviaoSettings
from mcp_gaviao_real.engines.base import ExtractedPage
from mcp_gaviao_real.models import ExecutionPolicy
from mcp_gaviao_real.utils.stdio_guard import redirect_stdout_to_stderr


def group_contiguous_ranges(page_nums: list[int]) -> list[tuple[int, int]]:
    """Group sorted 1-based page numbers into contiguous (start, end) ranges."""
    if not page_nums:
        return []
    ordered = sorted(set(page_nums))
    ranges: list[tuple[int, int]] = []
    start = ordered[0]
    prev = ordered[0]
    for p in ordered[1:]:
        if p == prev + 1:
            prev = p
        else:
            ranges.append((start, prev))
            start = p
            prev = p
    ranges.append((start, prev))
    return ranges


class DoclingEngine:
    """Lazy-loaded Docling v2 converter tuned by ExecutionPolicy with VRAM TTL unload."""

    def __init__(self, settings: GaviaoSettings, policy: ExecutionPolicy) -> None:
        self.settings = settings
        self.policy = policy
        self._converters: dict[bool, Any] = {}
        self._lock = threading.Lock()
        self._idle_timer: threading.Timer | None = None
        self._last_used: float = 0.0

    @property
    def is_loaded(self) -> bool:
        return bool(self._converters)

    def _cancel_timer(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _schedule_idle_unload(self) -> None:
        self._cancel_timer()
        ttl = self.policy.vram_idle_unload_s
        if ttl <= 0:
            return
        timer = threading.Timer(float(ttl), self.unload)
        timer.daemon = True
        timer.start()
        self._idle_timer = timer

    def unload(self) -> None:
        """Release cached Docling pipelines and free CUDA VRAM if loaded."""
        with self._lock:
            self._cancel_timer()
            if not self._converters:
                return
            self._converters.clear()
        gc.collect()
        if "torch" in sys.modules:
            torch = sys.modules["torch"]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _resolve_accelerator_device(self) -> Any:
        from docling.datamodel.accelerator_options import AcceleratorDevice

        dev = self.policy.device
        if dev == "cuda":
            if "torch" not in sys.modules:
                import torch  # noqa: F401
            torch_mod = sys.modules["torch"]
            if torch_mod.cuda.is_available():
                return AcceleratorDevice.CUDA
            return AcceleratorDevice.CPU
        if dev == "mps":
            return AcceleratorDevice.MPS
        return AcceleratorDevice.CPU

    def _build_converter(self, do_ocr: bool) -> Any:
        # Keep HuggingFace weights inside our managed local cache dir when HF_HOME isn't explicitly set
        hf_dir = self.settings.cache_dir / "hf_cache"
        hf_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(hf_dir))
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        with redirect_stdout_to_stderr():
            from docling.datamodel.accelerator_options import AcceleratorOptions
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                RapidOcrOptions,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption

            accel_device = self._resolve_accelerator_device()
            batch = max(1, self.policy.docling_batch_size)

            ocr_kwargs: dict[str, Any] = {}
            fields = RapidOcrOptions.model_fields
            if "mode" in fields:
                ocr_kwargs["mode"] = "full_page" if do_ocr else "default"
            elif "force_full_page_ocr" in fields:
                ocr_kwargs["force_full_page_ocr"] = do_ocr
            if "replace_broken_text_layer" in fields and do_ocr:
                ocr_kwargs["replace_broken_text_layer"] = True
            if "lang" in fields and self.settings.ocr_langs:
                ocr_kwargs["lang"] = self.settings.ocr_langs

            ocr_options = RapidOcrOptions(**ocr_kwargs)
            pipeline_options = PdfPipelineOptions(
                do_ocr=do_ocr,
                do_table_structure=True,
                ocr_options=ocr_options,
                accelerator_options=AcceleratorOptions(
                    num_threads=max(1, self.policy.omp_threads),
                    device=accel_device,
                ),
                ocr_batch_size=batch,
                layout_batch_size=batch,
                table_batch_size=batch,
            )
            return DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                }
            )

    def _get_converter(self, do_ocr: bool) -> Any:
        with self._lock:
            self._cancel_timer()
            self._last_used = time.monotonic()
            conv = self._converters.get(do_ocr)
            if conv is None:
                conv = self._build_converter(do_ocr)
                self._converters[do_ocr] = conv
            return conv

    @staticmethod
    def _tables_with_spans_html(docling_doc: Any, page_num: int) -> list[str]:
        """Export compact HTML <table> blocks for tables on page_num that have merged spans."""
        html_blocks: list[str] = []
        tables = getattr(docling_doc, "tables", None) or []
        for tbl in tables:
            provs = getattr(tbl, "prov", None) or []
            if provs and not any(getattr(pr, "page_no", None) == page_num for pr in provs):
                continue
            data = getattr(tbl, "data", None)
            cells = getattr(data, "table_cells", None) or []
            has_span = any(
                getattr(c, "col_span", 1) > 1 or getattr(c, "row_span", 1) > 1 for c in cells
            )
            if has_span and hasattr(tbl, "export_to_html"):
                try:
                    html = str(tbl.export_to_html(doc=docling_doc)).strip()
                    if html:
                        html_blocks.append(html)
                except Exception:
                    continue
        return html_blocks

    def extract_pages(
        self,
        path: Path,
        page_nums: list[int],
        *,
        doc_id: str,
        do_ocr: bool = False,
    ) -> list[ExtractedPage]:
        """Extract requested 1-based pages using Docling v2 (PDF ranges or Office)."""
        if not page_nums:
            return []

        engine_name = "docling_ocr" if do_ocr else "docling_layout"
        converter = self._get_converter(do_ocr)
        ext = path.suffix.lower()
        results: list[ExtractedPage] = []

        try:
            if ext == ".pdf":
                ranges = group_contiguous_ranges(page_nums)
                wanted = set(page_nums)
                for start, end in ranges:
                    with redirect_stdout_to_stderr():
                        conv_res = converter.convert(
                            str(path),
                            page_range=(start, end),
                        )
                    doc_obj = conv_res.document
                    for p in range(start, end + 1):
                        if p not in wanted:
                            continue
                        with redirect_stdout_to_stderr():
                            md = doc_obj.export_to_markdown(
                                page_no=p,
                                image_placeholder="",
                            )
                        span_html = self._tables_with_spans_html(doc_obj, p)
                        if span_html:
                            md = (md + "\n\n" + "\n\n".join(span_html)).strip()
                        cleaned = re.sub(r"\n{3,}", "\n\n", md).strip()
                        results.append(
                            ExtractedPage(
                                page_num=p,
                                engine=engine_name,
                                markdown=cleaned,
                            )
                        )
            else:
                with redirect_stdout_to_stderr():
                    conv_res = converter.convert(str(path))
                    md = conv_res.document.export_to_markdown(image_placeholder="")
                cleaned = re.sub(r"\n{3,}", "\n\n", md).strip()
                results.append(
                    ExtractedPage(
                        page_num=1,
                        engine=engine_name,
                        markdown=cleaned,
                    )
                )
        finally:
            if self.policy.clear_vram_after_job:
                self.unload()
            else:
                self._schedule_idle_unload()

        return results
