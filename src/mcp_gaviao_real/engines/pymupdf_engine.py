"""Fast CPU PDF extraction engine backed by PyMuPDF / pymupdf4llm."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from mcp_gaviao_real.engines.base import ExtractedPage
from mcp_gaviao_real.utils.stdio_guard import redirect_stdout_to_stderr


class PyMuPdfEngine:
    """Extract clean born-digital PDF pages rapidly using pymupdf4llm."""

    name = "pymupdf_fast"

    def extract_pages(
        self,
        path: Path,
        page_nums: list[int],
        *,
        doc_id: str,
    ) -> list[ExtractedPage]:
        if not page_nums:
            return []

        with redirect_stdout_to_stderr():
            import pymupdf4llm

        zero_based = [p - 1 for p in page_nums]
        results: list[ExtractedPage] = []

        with redirect_stdout_to_stderr():
            doc = pymupdf.open(path)
        try:
            chunks_by_zero_idx: dict[int, str] = {}
            try:
                with redirect_stdout_to_stderr():
                    chunks = pymupdf4llm.to_markdown(
                        doc,
                        pages=zero_based,
                        page_chunks=True,
                        write_images=False,
                    )
                if isinstance(chunks, list):
                    for pos, chunk in enumerate(chunks):
                        meta = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
                        raw_p = meta.get("page")
                        if isinstance(raw_p, int) and raw_p >= 1:
                            zero_idx = raw_p - 1
                        elif pos < len(zero_based):
                            zero_idx = zero_based[pos]
                        else:
                            continue
                        text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                        chunks_by_zero_idx[zero_idx] = text
            except Exception:
                # Fall back page-by-page below
                chunks_by_zero_idx = {}

            for p_num in page_nums:
                z_idx = p_num - 1
                md = chunks_by_zero_idx.get(z_idx)
                warnings: list[str] = []
                if md is None:
                    try:
                        with redirect_stdout_to_stderr():
                            single = pymupdf4llm.to_markdown(
                                doc,
                                pages=[z_idx],
                                page_chunks=False,
                                write_images=False,
                            )
                        md = single if isinstance(single, str) else str(single)
                    except Exception as exc:
                        md = doc[z_idx].get_text("text") or ""
                        warnings.append(
                            f"Página {p_num}: fallback para texto simples PyMuPDF ({exc})."
                        )

                cleaned = re.sub(r"\n{3,}", "\n\n", md).strip()
                results.append(
                    ExtractedPage(
                        page_num=p_num,
                        engine=self.name,
                        markdown=cleaned,
                        warnings=warnings,
                    )
                )
        finally:
            doc.close()

        return results
