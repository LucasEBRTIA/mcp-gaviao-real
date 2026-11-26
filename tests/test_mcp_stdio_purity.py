"""End-to-end MCP stdio integration test ensuring clean JSON-RPC channel."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


@pytest.mark.asyncio
async def test_mcp_stdio_end_to_end_purity(
    tmp_path: Path,
    mixed_pdf_path: Path,
) -> None:
    env = os.environ.copy()
    env["GAVIAO_CACHE_DIR"] = str(tmp_path / "mcp_cache")
    env["PYTHONIOENCODING"] = "utf-8"

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_gaviao_real.cli", "serve", "--transport", "stdio"],
        env=env,
    )

    async with (
        stdio_client(server_params) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()

        tools_res = await session.list_tools()
        tool_names = {t.name for t in tools_res.tools}
        assert tool_names == {
            "inspect_document",
            "read_document",
            "get_document_image",
            "search_document",
        }
        for t in tools_res.tools:
            assert t.annotations is not None
            assert t.annotations.read_only_hint is True

        # 1. inspect_document
        insp_res = await session.call_tool("inspect_document", {"source": str(mixed_pdf_path)})
        assert not insp_res.is_error
        insp_data = json.loads(insp_res.content[0].text)
        doc_id = insp_data["doc_id"]
        assert insp_data["page_count"] == 6

        # 2. search_document
        search_res = await session.call_tool(
            "search_document",
            {"source": doc_id, "query": "relatorio liquido", "mode": "hybrid"},
        )
        assert not search_res.is_error
        search_data = json.loads(search_res.content[0].text)
        assert search_data["total_hits"] >= 1

        # 3. read_document
        read_res = await session.call_tool(
            "read_document",
            {"source": doc_id, "pages": "1-2", "include_images": True},
        )
        assert not read_res.is_error
        read_data = json.loads(read_res.content[0].text)
        assert read_data["pages_returned"] == [1, 2]
        assert len(read_data["images"]) == 1
        fig_id = read_data["images"][0]["image_id"]

        # 4. get_document_image (extracted figure + page:3 render + zoom crop)
        img_res = await session.call_tool(
            "get_document_image",
            {"source": doc_id, "target": fig_id, "max_size": 600},
        )
        assert not img_res.is_error
        assert len(img_res.content) == 2
        assert img_res.content[1].type == "image"
        assert img_res.content[1].mime_type == "image/webp"

        zoom_res = await session.call_tool(
            "get_document_image",
            {
                "source": doc_id,
                "target": "page:5",
                "crop_bbox": [100, 150, 900, 600],
            },
        )
        assert not zoom_res.is_error
        assert zoom_res.content[1].type == "image"
