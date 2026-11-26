"""Pydantic v2 domain models and MCP structured output schemas."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class HardwareTier(str, Enum):
    GPU_HIGH = "gpu_high"
    GPU_LOW = "gpu_low"
    CPU_RICH = "cpu_rich"
    LITE = "lite"


class PageClass(str, Enum):
    DIGITAL_CLEAN = "digital_clean"
    COMPLEX_LAYOUT = "complex_layout"
    SCANNED = "scanned"
    CORRUPTED_FONT = "corrupted_font"
    VISUAL_HEAVY = "visual_heavy"


class HardwareProfile(BaseModel):
    os_name: str
    arch: str
    cpu_cores: int
    ram_total_gb: float
    ram_available_gb: float
    gpu_vendor: Literal["nvidia", "apple"] | None = None
    gpu_name: str | None = None
    vram_total_mb: int | None = None
    has_docling: bool = False
    has_torch: bool = False
    has_rapidocr: bool = False
    has_vlm: bool = False


class ExecutionPolicy(BaseModel):
    tier: HardwareTier
    device: Literal["cuda", "mps", "cpu"]
    can_use_docling: bool
    docling_batch_size: int
    omp_threads: int
    max_ocr_pages_per_call: int
    clear_vram_after_job: bool
    vram_idle_unload_s: int
    recommended_install_cmd: str
    notes: list[str] = Field(default_factory=list)


class TocEntry(BaseModel):
    level: int
    title: str
    page: int


class ImageRef(BaseModel):
    image_id: str
    page: int
    kind: str
    width: int
    height: int
    uri: str
    bbox: list[float] | None = None
    caption: str | None = None
    vlm_description: str | None = None


class PageProfile(BaseModel):
    page: int
    classification: PageClass
    char_count: int
    image_count: int
    image_coverage: float
    garbled_ratio: float
    table_hint: bool
    suggest_page_render: bool
    estimated_tokens: int


class DocumentInspection(BaseModel):
    doc_id: str
    path: str
    format: str
    file_size_bytes: int
    page_count: int
    estimated_total_tokens: int
    toc: list[TocEntry] = Field(default_factory=list)
    pages: list[PageProfile] = Field(default_factory=list)
    images: list[ImageRef] = Field(default_factory=list)
    active_tier: HardwareTier
    recommended_strategy: str


class ReadDocumentResult(BaseModel):
    doc_id: str
    path: str
    pages_returned: list[int]
    engines_used: dict[str, list[int]] = Field(default_factory=dict)
    markdown: str
    images: list[ImageRef] = Field(default_factory=list)
    estimated_tokens: int
    truncated: bool = False
    next_page: int | None = None
    warnings: list[str] = Field(default_factory=list)


class SearchHit(BaseModel):
    page: int
    score: float
    breadcrumbs: str
    snippet: str


class SearchResponse(BaseModel):
    doc_id: str
    query: str
    mode: str
    total_hits: int
    hits: list[SearchHit] = Field(default_factory=list)
    unindexed_scanned_pages: list[int] = Field(default_factory=list)
