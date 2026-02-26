from __future__ import annotations

from pathlib import Path

# Project root and common data path.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

# Default model and eval settings.
DEFAULT_MODEL_NAME = "all-MiniLM-L12-v2"
DEFAULT_EVAL_K = 10

