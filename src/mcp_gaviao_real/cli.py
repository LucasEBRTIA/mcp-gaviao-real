"""CLI entrypoint for mcp-gaviao-real (serve, doctor, setup, cache)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from mcp_gaviao_real.config import GaviaoSettings
from mcp_gaviao_real.hardware import detect_hardware, resolve_policy
from mcp_gaviao_real.models import HardwareProfile
from mcp_gaviao_real.storage.cache import DocumentCache
from mcp_gaviao_real.utils.stdio_guard import configure_stderr_logging


def recommended_sync_args(
    profile: HardwareProfile,
    *,
    lite: bool = False,
) -> list[str]:
    """Return the optimal `uv sync` command arguments for a given HardwareProfile."""
    if lite:
        return ["uv", "sync"]
    if profile.gpu_vendor == "nvidia":
        return ["uv", "sync", "--extra", "accurate", "--extra", "cuda"]
    if profile.gpu_vendor == "apple":
        return ["uv", "sync", "--extra", "accurate"]
    if profile.ram_total_gb >= 12.0:
        return ["uv", "sync", "--extra", "accurate", "--extra", "cpu"]
    return ["uv", "sync"]


def build_mcp_config_snippet(
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8765,
) -> dict[str, object]:
    """Build a ready-to-paste `mcpServers` JSON snippet for MCP clients."""
    if transport == "streamable-http":
        return {
            "mcpServers": {
                "mcp-gaviao-real": {
                    "serverUrl": f"http://{host}:{port}/mcp",
                }
            }
        }
    return {
        "mcpServers": {
            "mcp-gaviao-real": {
                "command": sys.executable,
                "args": ["-m", "mcp_gaviao_real.cli", "serve", "--transport", "stdio"],
                "env": {"PYTHONIOENCODING": "utf-8"},
                "disabled": False,
            }
        }
    }


def _cmd_serve(args: argparse.Namespace) -> int:
    from mcp_gaviao_real.utils.stdio_guard import redirect_stdout_to_stderr

    configure_stderr_logging()
    with redirect_stdout_to_stderr():
        from mcp_gaviao_real.server import mcp

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(
            transport="streamable-http",
            host=args.host,
            port=args.port,
        )
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    settings = GaviaoSettings()
    profile = detect_hardware(settings)
    policy = resolve_policy(profile, settings)
    cache = DocumentCache(settings.cache_dir, max_cache_bytes=settings.max_cache_bytes)
    try:
        cache_stats = cache.stats()
    finally:
        cache.close()

    report = {
        "hardware": profile.model_dump(),
        "policy": policy.model_dump(),
        "cache": cache_stats,
        "vlm_configured": profile.has_vlm,
    }
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    print("=== mcp-gaviao-real doctor ===")
    print(f"Sistema      : {profile.os_name} ({profile.arch}) | {profile.cpu_cores} cores")
    print(f"RAM          : {profile.ram_total_gb} GB total ({profile.ram_available_gb} GB livre)")
    if profile.gpu_name:
        vram = f" ({profile.vram_total_mb} MiB VRAM)" if profile.vram_total_mb else ""
        print(f"GPU          : {profile.gpu_name}{vram}")
    else:
        print("GPU          : Nenhuma detectada (modo CPU)")
    print(
        f"Extras       : docling={profile.has_docling} | torch={profile.has_torch} | rapidocr={profile.has_rapidocr}"
    )
    print(
        f"Perfil ativo : {policy.tier.value} (device={policy.device}, batch={policy.docling_batch_size}, threads={policy.omp_threads})"
    )
    print(
        f"Cache        : {cache_stats['cache_dir']} ({cache_stats['documents']} docs, {cache_stats['cached_pages']} págs)"
    )
    print(f"Recomendação : {policy.recommended_install_cmd}")
    for note in policy.notes:
        print(f"Nota         : {note}")
    return 0


def _cmd_setup(args: argparse.Namespace) -> int:
    settings = GaviaoSettings()
    profile = detect_hardware(settings)
    sync_cmd = recommended_sync_args(profile, lite=args.lite)
    snippet = build_mcp_config_snippet(
        transport=args.transport,
        host=args.host,
        port=args.port,
    )

    if args.print_mcp_config:
        print(json.dumps(snippet, indent=2, ensure_ascii=False))
        return 0

    print("=== mcp-gaviao-real setup ===")
    gpu_desc = profile.gpu_name or "Nenhuma (CPU-only)"
    if profile.vram_total_mb:
        gpu_desc += f" ({profile.vram_total_mb} MiB)"
    print(
        f"Máquina      : {profile.os_name} ({profile.arch}) | {profile.cpu_cores} cores | {profile.ram_total_gb} GB RAM | GPU: {gpu_desc}"
    )
    print(f"Comando sync : {' '.join(sync_cmd)}")
    print("\nConfiguração MCP (mcp_config.json):")
    print(json.dumps(snippet, indent=2, ensure_ascii=False))

    if args.apply:
        print(f"\nExecutando: {' '.join(sync_cmd)}")
        proc = subprocess.run(sync_cmd, check=False)
        return int(proc.returncode)
    return 0


def _cmd_cache(args: argparse.Namespace) -> int:
    settings = GaviaoSettings()
    cache = DocumentCache(settings.cache_dir, max_cache_bytes=settings.max_cache_bytes)
    try:
        if args.cache_action == "stats":
            print(json.dumps(cache.stats(), indent=2, ensure_ascii=False))
        elif args.cache_action == "clear":
            removed = cache.clear()
            print(f"Cache limpo ({removed} documento(s) removido(s)).")
    finally:
        cache.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-gaviao-real",
        description="Servidor MCP híbrido de extração de documentos.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Inicia o servidor MCP.")
    p_serve.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
    )
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8765)
    p_serve.set_defaults(func=_cmd_serve)

    p_doc = sub.add_parser("doctor", help="Diagnostica hardware, motores e cache.")
    p_doc.add_argument("--json", action="store_true", help="Saída em JSON.")
    p_doc.set_defaults(func=_cmd_doctor)

    p_setup = sub.add_parser(
        "setup",
        help="Detecta a máquina, recomenda/executa `uv sync` e gera config MCP.",
    )
    p_setup.add_argument(
        "--apply",
        action="store_true",
        help="Executa automaticamente o comando `uv sync` recomendado.",
    )
    p_setup.add_argument(
        "--lite",
        action="store_true",
        help="Força instalação leve sem Docling/PyTorch.",
    )
    p_setup.add_argument(
        "--print-mcp-config",
        action="store_true",
        help="Imprime apenas o JSON para `mcp_config.json`.",
    )
    p_setup.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
    )
    p_setup.add_argument("--host", default="127.0.0.1")
    p_setup.add_argument("--port", type=int, default=8765)
    p_setup.set_defaults(func=_cmd_setup)

    p_cache = sub.add_parser("cache", help="Gerencia o cache local de documentos.")
    cache_sub = p_cache.add_subparsers(dest="cache_action", required=True)
    cache_sub.add_parser("stats", help="Exibe estatísticas do cache.")
    cache_sub.add_parser("clear", help="Limpa todo o cache.")
    p_cache.set_defaults(func=_cmd_cache)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
