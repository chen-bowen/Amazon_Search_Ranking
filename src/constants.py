from __future__ import annotations

from pathlib import Path

# Project root and common data/checkpoint paths.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
# Directory for trained model checkpoints (separate from raw/processed data).
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
# Cache directory for downloaded pretrained weights.
MODEL_CACHE_DIR = DATA_DIR / ".model_cache"

# Default model and eval settings.
DEFAULT_MODEL_NAME = "all-MiniLM-L12-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
DEFAULT_EVAL_K = 10

# ESCI label -> gain for nDCG (paper: E=1.0, S=0.1, C=0.01, I=0.0)
ESCI_LABEL2GAIN = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}

# ESCI label -> class index for multi-task learning Task 2 (4-way E/S/C/I classification). Order: E, S, C, I.
ESCI_LABEL2ID = {"E": 0, "S": 1, "C": 2, "I": 3}
ESCI_ID2LABEL = ["E", "S", "C", "I"]

# Default config for multi-task reranker training (fallback when YAML is missing).
MULTI_TASK_RERANKER_DEFAULTS = {
    "data_dir": "data",
    "model_name": DEFAULT_RERANKER_MODEL,
    "product_col": "product_text",
    "save_path": "checkpoints/multi_task_reranker",
    "val_frac": 0.1,
    "epochs": 1,
    "batch_size": 16,
    "max_length": 512,
    "lr": 7e-6,
    "warmup_steps": 5000,
    "task_weight_ranking": 1.0,
    "task_weight_esci": 0.5,
    "task_weight_substitute": 0.5,
    "evaluation_steps": 15000,
    "eval_max_queries": None,
    "recall_at": 10,
}

# Default config for reranker inference (fallback when YAML is missing).
INFER_RERANKER_DEFAULTS = {
    "model_path": "checkpoints/reranker",
    "data_dir": "data",
    "product_col": "product_text",
    "batch_size": 16,
    "top_k": 5,
    "query_index": 0,
    "small_version": False,
}
