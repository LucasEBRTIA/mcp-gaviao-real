"""Configuration settings and path validation for mcp-gaviao-real."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from platformdirs import user_cache_dir
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_cache_dir() -> Path:
    return Path(user_cache_dir("mcp-gaviao-real", appauthor=False)) / "cache"


class GaviaoSettings(BaseSettings):
    """Runtime settings loaded from environment variables prefixed with GAVIAO_."""

    model_config = SettingsConfigDict(
        env_prefix="GAVIAO_",
        env_file=".env",
        extra="ignore",
    )

    cache_dir: Path = Field(default_factory=_default_cache_dir)
    max_cache_bytes: int = 2 * 1024 * 1024 * 1024  # 2 GiB
    allowed_dirs: list[Path] = Field(default_factory=list)

    profile: Literal["auto", "gpu_high", "gpu_low", "cpu_rich", "lite"] = "auto"
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    gpu_idle_unload_seconds: int = 300
    ocr_langs: list[str] = Field(default_factory=lambda: ["pt", "en"])

    default_max_tokens: int = 12000
    default_image_max_size: int = 1280
    max_xlsx_rows_per_sheet: int = 200

    vlm_base_url: str | None = None
    vlm_api_key: str | None = None
    vlm_model: str | None = None
    vlm_concurrency: int = 3
    vlm_timeout_seconds: float = 45.0


def validate_file_path(path: str | Path, settings: GaviaoSettings) -> Path:
    """Resolve and validate that a path points to an existing readable file within allowed_dirs."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"O caminho não é um arquivo regular: {resolved}")

    if settings.allowed_dirs:
        allowed_roots = [d.expanduser().resolve() for d in settings.allowed_dirs]
        in_allowed = False
        for root in allowed_roots:
            try:
                resolved.relative_to(root)
                in_allowed = True
                break
            except ValueError:
                continue
        if not in_allowed:
            raise PermissionError(
                f"Acesso negado: '{resolved}' está fora dos diretórios permitidos (GAVIAO_ALLOWED_DIRS)."
            )

    return resolved
