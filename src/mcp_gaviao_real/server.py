"""FastMCP / MCPServer definition exposing 4 tools and 2 resource templates."""

from __future__ import annotations

import base64
import json
from typing import Annotated, Literal

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ImageContent, TextContent, ToolAnnotations
from pydantic import Field

from mcp_gaviao_real.config import GaviaoSettings
from mcp_gaviao_real.models import (
    DocumentInspection,
    ReadDocumentResult,
    SearchResponse,
)
from mcp_gaviao_real.router import DocumentRouter
from mcp_gaviao_real.utils.stdio_guard import configure_stderr_logging

READ_ONLY_ANNOTATIONS = ToolAnnotations(
    read_only_hint=True,
    idempotent_hint=True,
    destructive_hint=False,
    open_world_hint=False,
)

mcp = MCPServer(
    name="mcp-gaviao-real",
    instructions=(
        "Servidor de extração inteligente de documentos. "
        "Prefira sempre chamar `inspect_document` ou `search_document` antes de ler páginas "
        "com `read_document` (em intervalos curtos como '1-5'). "
        "Use `get_document_image` apenas sob demanda para figuras relevantes, páginas visuais "
        "(`target='page:N'`) ou zoom de alta resolução (`crop_bbox`)."
    ),
)

_router: DocumentRouter | None = None


def get_router() -> DocumentRouter:
    """Return lazy singleton DocumentRouter initialized on first tool/resource call."""
    global _router
    if _router is None:
        configure_stderr_logging()
        _router = DocumentRouter(GaviaoSettings())
    return _router


@mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
def inspect_document(
    source: Annotated[
        str,
        Field(
            description="Caminho do arquivo (PDF, DOCX, PPTX, XLSX, HTML, CSV) ou doc_id retornado anteriormente."
        ),
    ],
) -> DocumentInspection:
    """Inspeciona rapidamente estrutura, TOC, tokens estimados, classificação por página e inventário de imagens."""
    return get_router().inspect(source)


@mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
async def read_document(
    source: Annotated[
        str,
        Field(description="Caminho do arquivo ou doc_id retornado por inspect_document."),
    ],
    pages: Annotated[
        str,
        Field(
            description="Intervalo 1-based de páginas/abas/slides, ex.: '1-5', '1,3,8-10' ou 'all'."
        ),
    ] = "1-5",
    mode: Annotated[
        Literal["auto", "fast", "accurate", "ocr"],
        Field(
            description="Estratégia de extração: auto (híbrido por página), fast, accurate ou ocr."
        ),
    ] = "auto",
    include_images: Annotated[
        bool,
        Field(description="Inclui lista detalhada de metadados de imagens no JSON estruturado."),
    ] = False,
    vlm_captions: Annotated[
        bool,
        Field(description="Gera descrições textuais via endpoint VLM configurado (opcional)."),
    ] = False,
    max_tokens: Annotated[
        int,
        Field(ge=200, le=100000, description="Orçamento máximo aproximado de tokens na resposta."),
    ] = 12000,
    ctx: Context | None = None,
) -> ReadDocumentResult:
    """Extrai Markdown estruturado do intervalo solicitado com cache granular, breadcrumbs e placeholders de imagem."""
    progress_cb = ctx.report_progress if ctx is not None else None
    return await get_router().read(
        source=source,
        pages=pages,
        mode=mode,
        include_images=include_images,
        vlm_captions=vlm_captions,
        max_tokens=max_tokens,
        progress_cb=progress_cb,
    )


@mcp.tool(annotations=READ_ONLY_ANNOTATIONS, structured_output=False)
def get_document_image(
    source: Annotated[
        str,
        Field(description="Caminho do arquivo ou doc_id."),
    ],
    target: Annotated[
        str,
        Field(description="ID da figura (ex.: 'p3_fig1') ou página inteira (ex.: 'page:3')."),
    ],
    max_size: Annotated[
        int,
        Field(
            ge=128, le=4096, description="Dimensão máxima em pixels no maior eixo (padrão 1280)."
        ),
    ] = 1280,
    crop_bbox: Annotated[
        list[int] | None,
        Field(
            description="Recorte opcional em 300 DPI [x0, y0, x1, y1] normalizado 0..1000 (somente com target='page:N')."
        ),
    ] = None,
) -> list[TextContent | ImageContent]:
    """Retorna imagem otimizada (WebP) de uma figura extraída, página inteira ou recorte de zoom 300 DPI."""
    meta, webp_bytes = get_router().get_image(
        source=source,
        target=target,
        max_size=max_size,
        crop_bbox=crop_bbox,
    )
    b64_data = base64.b64encode(webp_bytes).decode("ascii")
    return [
        TextContent(type="text", text=json.dumps(meta, ensure_ascii=False)),
        ImageContent(type="image", data=b64_data, mime_type="image/webp"),
    ]


@mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
def search_document(
    source: Annotated[
        str,
        Field(description="Caminho do arquivo ou doc_id."),
    ],
    query: Annotated[
        str,
        Field(description="Termo de busca ou expressão regular."),
    ],
    mode: Annotated[
        Literal["fts", "regex", "hybrid"],
        Field(description="Modo de busca: fts (BM25 sem acentos), regex ou hybrid."),
    ] = "hybrid",
    limit: Annotated[
        int,
        Field(ge=1, le=50, description="Número máximo de páginas retornadas."),
    ] = 10,
) -> SearchResponse:
    """Busca textual/regex rápida via SQLite FTS5 retornando páginas, breadcrumbs e trechos."""
    return get_router().search(
        source=source,
        query=query,
        mode=mode,
        limit=limit,
    )


@mcp.resource(
    "doc://{doc_id}/page/{page_num}",
    mime_type="text/markdown",
    description="Conteúdo Markdown extraído de uma página específica pelo doc_id.",
)
async def resource_document_page(doc_id: str, page_num: str) -> str:
    res = await get_router().read(source=doc_id, pages=page_num, mode="auto")
    return res.markdown


@mcp.resource(
    "doc://{doc_id}/img/{image_id}",
    mime_type="image/webp",
    description="Blob WebP otimizado de uma figura extraída pelo doc_id e image_id.",
)
def resource_document_image(doc_id: str, image_id: str) -> bytes:
    _, webp_bytes = get_router().get_image(source=doc_id, target=image_id)
    return webp_bytes
