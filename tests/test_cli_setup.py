"""Unit tests for `mcp-gaviao-real setup` CLI helpers and JSON output."""

from __future__ import annotations

import json

import pytest

from mcp_gaviao_real.cli import (
    build_mcp_config_snippet,
    main,
    recommended_sync_args,
)
from mcp_gaviao_real.models import HardwareProfile


def test_recommended_sync_args_all_machines() -> None:
    nvidia = HardwareProfile(
        os_name="Windows",
        arch="AMD64",
        cpu_cores=14,
        ram_total_gb=31.6,
        ram_available_gb=12.0,
        gpu_vendor="nvidia",
        gpu_name="RTX 4060 Ti",
        vram_total_mb=16380,
    )
    assert recommended_sync_args(nvidia) == [
        "uv",
        "sync",
        "--extra",
        "accurate",
        "--extra",
        "cuda",
    ]
    assert recommended_sync_args(nvidia, lite=True) == ["uv", "sync"]

    apple = HardwareProfile(
        os_name="Darwin",
        arch="arm64",
        cpu_cores=10,
        ram_total_gb=16.0,
        ram_available_gb=8.0,
        gpu_vendor="apple",
        gpu_name="Apple Silicon (MPS)",
    )
    assert recommended_sync_args(apple) == ["uv", "sync", "--extra", "accurate"]

    cpu_rich = HardwareProfile(
        os_name="Linux",
        arch="x86_64",
        cpu_cores=8,
        ram_total_gb=16.0,
        ram_available_gb=10.0,
    )
    assert recommended_sync_args(cpu_rich) == [
        "uv",
        "sync",
        "--extra",
        "accurate",
        "--extra",
        "cpu",
    ]

    low_ram = HardwareProfile(
        os_name="Linux",
        arch="x86_64",
        cpu_cores=4,
        ram_total_gb=6.0,
        ram_available_gb=2.0,
    )
    assert recommended_sync_args(low_ram) == ["uv", "sync"]


def test_build_mcp_config_snippet() -> None:
    stdio_cfg = build_mcp_config_snippet("stdio")
    entry = stdio_cfg["mcpServers"]["mcp-gaviao-real"]
    assert entry["args"] == ["-m", "mcp_gaviao_real.cli", "serve", "--transport", "stdio"]
    assert entry["disabled"] is False

    http_cfg = build_mcp_config_snippet("streamable-http", host="0.0.0.0", port=9000)
    assert http_cfg["mcpServers"]["mcp-gaviao-real"]["serverUrl"] == "http://0.0.0.0:9000/mcp"


def test_cli_setup_print_mcp_config(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["setup", "--print-mcp-config"])
    assert rc == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert "mcp-gaviao-real" in parsed["mcpServers"]
