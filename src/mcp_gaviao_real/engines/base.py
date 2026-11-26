"""Base extraction engine contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from mcp_gaviao_real.models import ImageRef


@dataclass
class ExtractedPage:
    """Single extracted logical page produced by an ExtractionEngine."""

    page_num: int
    engine: str
    markdown: str
    breadcrumbs: str = ""
    images: list[ImageRef] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ExtractionEngine(Protocol):
    """Protocol implemented by all extraction backends."""

    name: str

    def extract_pages(
        self,
        path: Path,
        page_nums: list[int],
        *,
        doc_id: str,
    ) -> list[ExtractedPage]: ...
