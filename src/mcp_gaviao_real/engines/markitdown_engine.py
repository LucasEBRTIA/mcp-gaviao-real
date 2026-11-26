"""Lightweight Office, HTML, CSV, and text extraction engine via MarkItDown + openpyxl."""

from __future__ import annotations

from pathlib import Path

import openpyxl
from markitdown import MarkItDown

from mcp_gaviao_real.config import GaviaoSettings
from mcp_gaviao_real.engines.base import ExtractedPage
from mcp_gaviao_real.utils.stdio_guard import redirect_stdout_to_stderr


class MarkItDownEngine:
    """Fast CPU extraction for DOCX, PPTX, XLSX, HTML, CSV, MD, and TXT."""

    name = "markitdown_fast"

    def __init__(self, settings: GaviaoSettings) -> None:
        self.settings = settings
        self._converter: MarkItDown | None = None

    def _get_converter(self) -> MarkItDown:
        if self._converter is None:
            with redirect_stdout_to_stderr():
                self._converter = MarkItDown()
        return self._converter

    def extract_pages(
        self,
        path: Path,
        page_nums: list[int],
        *,
        doc_id: str,
    ) -> list[ExtractedPage]:
        if not page_nums:
            return []

        ext = path.suffix.lower()
        if ext in {".xlsx", ".xlsm"}:
            return self._extract_xlsx(path, page_nums)
        if ext == ".pptx":
            return self._extract_pptx(path, page_nums)
        if ext in {".txt", ".md", ".csv", ".tsv"}:
            return self._extract_plain_or_csv(path, page_nums, ext)

        with redirect_stdout_to_stderr():
            res = self._get_converter().convert(str(path))
        md = (res.text_content or "").strip()
        return [
            ExtractedPage(
                page_num=1,
                engine=self.name,
                markdown=md,
            )
            for p in page_nums
            if p == 1
        ]

    def _extract_xlsx(self, path: Path, page_nums: list[int]) -> list[ExtractedPage]:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        max_rows = self.settings.max_xlsx_rows_per_sheet
        pages: list[ExtractedPage] = []
        try:
            sheet_names = wb.sheetnames
            for p_num in page_nums:
                if p_num < 1 or p_num > len(sheet_names):
                    continue
                name = sheet_names[p_num - 1]
                ws = wb[name]
                rows_md: list[str] = []
                total_rows = ws.max_row or 0
                headers: list[str] | None = None
                captured = 0

                for row in ws.iter_rows(values_only=True):
                    cells = [
                        "" if v is None else str(v).replace("\n", " ").replace("|", "\\|").strip()
                        for v in row
                    ]
                    if not any(cells):
                        continue
                    if headers is None:
                        headers = cells
                        rows_md.append("| " + " | ".join(headers) + " |")
                        rows_md.append("| " + " | ".join("---" for _ in headers) + " |")
                    else:
                        if len(cells) < len(headers):
                            cells.extend([""] * (len(headers) - len(cells)))
                        elif len(cells) > len(headers):
                            cells = cells[: len(headers)]
                        rows_md.append("| " + " | ".join(cells) + " |")
                    captured += 1
                    if captured >= max_rows:
                        break

                md = f"## Planilha: {name}\n\n" + (
                    "\n".join(rows_md) if rows_md else "_(Aba vazia)_"
                )
                warnings: list[str] = []
                if total_rows > max_rows:
                    notice = f"Aba '{name}' limitada às primeiras {max_rows} linhas de {total_rows} totais."
                    md += f"\n\n> **Nota:** {notice}"
                    warnings.append(notice)

                pages.append(
                    ExtractedPage(
                        page_num=p_num,
                        engine=self.name,
                        markdown=md,
                        breadcrumbs=name,
                        warnings=warnings,
                    )
                )
        finally:
            wb.close()
        return pages

    def _extract_pptx(self, path: Path, page_nums: list[int]) -> list[ExtractedPage]:
        try:
            from pptx import Presentation

            prs = Presentation(str(path))
            pages: list[ExtractedPage] = []
            for p_num in page_nums:
                if p_num < 1 or p_num > len(prs.slides):
                    continue
                slide = prs.slides[p_num - 1]
                lines: list[str] = [f"## Slide {p_num}"]
                title_text = ""
                for shape in slide.shapes:
                    if getattr(shape, "has_text_frame", False):
                        txt = shape.text.strip()
                        if txt:
                            if not title_text:
                                title_text = txt.splitlines()[0][:80]
                            lines.append(txt)
                    if getattr(shape, "has_table", False):
                        tbl = shape.table
                        t_rows: list[str] = []
                        for r_idx, r in enumerate(tbl.rows):
                            vals = [
                                c.text.replace("\n", " ").replace("|", "\\|").strip()
                                for c in r.cells
                            ]
                            t_rows.append("| " + " | ".join(vals) + " |")
                            if r_idx == 0:
                                t_rows.append("| " + " | ".join("---" for _ in vals) + " |")
                        if t_rows:
                            lines.append("\n".join(t_rows))
                pages.append(
                    ExtractedPage(
                        page_num=p_num,
                        engine=self.name,
                        markdown="\n\n".join(lines),
                        breadcrumbs=title_text or f"Slide {p_num}",
                    )
                )
            return pages
        except Exception:
            with redirect_stdout_to_stderr():
                res = self._get_converter().convert(str(path))
            return [
                ExtractedPage(
                    page_num=1,
                    engine=self.name,
                    markdown=(res.text_content or "").strip(),
                )
            ]

    def _extract_plain_or_csv(
        self, path: Path, page_nums: list[int], ext: str
    ) -> list[ExtractedPage]:
        if 1 not in page_nums:
            return []
        if ext in {".csv", ".tsv"}:
            with redirect_stdout_to_stderr():
                res = self._get_converter().convert(str(path))
            return [
                ExtractedPage(
                    page_num=1,
                    engine=self.name,
                    markdown=(res.text_content or "").strip(),
                )
            ]
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        return [ExtractedPage(page_num=1, engine=self.name, markdown=text)]
