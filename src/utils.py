"""Shared logging and utility helpers for the Amazon Search Retrieval project."""

from __future__ import annotations

import gc
import logging

import torch


def clear_torch_cache() -> None:
    """Run gc and empty CUDA/MPS caches before eval to free memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def resolve_device(device: torch.device | str | None) -> torch.device:
    """Resolve device: cuda > mps > cpu when None."""
    if device is not None:
        return torch.device(device) if isinstance(device, str) else device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

_GRAY = "\033[90m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"
_LEVEL_COLORS = {
    "DEBUG": _GRAY,
    "INFO": _GREEN,
    "WARNING": _YELLOW,
    "ERROR": _RED,
}


class ColoredFormatter(logging.Formatter):
    """Formatter that colorizes level name and logger name, message-only output."""

    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname:5}{_RESET}"
        record.name = f"{_GRAY}{record.name}{_RESET}"
        return super().format(record)


def setup_colored_logging(
    level: int = logging.INFO,
    fmt: str = "%(message)s",
    quiet_loggers: list[str] | None = None,
) -> None:
    """Configure root logger with colored, compact output.

    Args:
        level: Root log level (default INFO).
        fmt: Log record format (default message only).
        quiet_loggers: Logger names to set to WARNING (e.g. httpx, urllib3, sentence_transformers).
    """
    if quiet_loggers:
        for name in quiet_loggers:
            logging.getLogger(name).setLevel(logging.WARNING)

    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter(fmt))

    logging.root.handlers.clear()
    logging.root.addHandler(handler)
    logging.root.setLevel(level)
