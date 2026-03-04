from __future__ import annotations

from pathlib import Path

# Project root and common data path.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
MODEL_CACHE_DIR = DATA_DIR / ".model_cache"

# Default model and eval settings.
DEFAULT_MODEL_NAME = "all-MiniLM-L12-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
DEFAULT_EVAL_K = 10

# ESCI label -> gain for nDCG (paper: E=1.0, S=0.1, C=0.01, I=0.0)
ESCI_LABEL2GAIN = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
