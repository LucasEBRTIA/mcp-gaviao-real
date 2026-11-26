"""Stdio channel protection so libraries never corrupt MCP JSON-RPC output."""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from typing import Iterator


def configure_stderr_logging(level: int = logging.WARNING) -> None:
    """Configure root logging to write strictly to sys.stderr."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


@contextmanager
def redirect_stdout_to_stderr() -> Iterator[None]:
    """Temporarily redirect Python sys.stdout to sys.stderr during library calls."""
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old_stdout
