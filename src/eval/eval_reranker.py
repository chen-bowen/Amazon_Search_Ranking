#!/usr/bin/env python3
"""
Evaluate trained ESCI reranker: compute nDCG, MRR, MAP, Recall@10 on test set.
Run: uv run python -m src.eval.eval_reranker [--config configs/reranker.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from src.constants import DATA_DIR, REPO_ROOT
from src.data.load_data import ESCIDataLoader
from src.eval.evaluator import ESCIMetricsEvaluator
from src.models.reranker import load_reranker
from src.utils import load_config

logger = logging.getLogger(__name__)

# Fallback values when config is missing or a key is absent
DEFAULTS = {
    "model_path": "checkpoints/reranker",
    "data_dir": str(DATA_DIR),
    "product_col": "product_text",
    "eval_max_queries": None,
    "recall_at": 10,
    "small_version": False,
}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Parse CLI: only --config to specify which YAML to load
    p = argparse.ArgumentParser(
        description="Evaluate ESCI reranker (nDCG, MRR, MAP, Recall@10)"
    )
    p.add_argument(
        "--config", default="configs/reranker.yaml", help="Path to YAML config"
    )
    args = p.parse_args()

    config_path = REPO_ROOT / args.config
    configs = load_config(config_path, DEFAULTS)

    test_df = _load_test_data(configs)
    if len(test_df) == 0:
        logger.error("No test data found.")
        return 1

    metrics = _run_evaluation(configs, test_df)
    recall_at = configs["recall_at"]
    logger.info("nDCG = %.4f", metrics["ndcg"])
    logger.info("MRR  = %.4f", metrics["mrr"])
    logger.info("MAP  = %.4f", metrics["map"])
    logger.info("Recall@%d = %.4f", recall_at, metrics["recall"])
    return 0


def _load_test_data(configs: dict) -> pd.DataFrame:
    """Load test data: prefer pre-saved parquet, else load from raw and split."""
    base = Path(configs["data_dir"])
    test_path = base / "esci_test.parquet"
    if test_path.exists():
        return pd.read_parquet(test_path)
    loader = ESCIDataLoader(
        data_dir=base,
        small_version=configs.get("small_version", False),
    )
    _, test_df = loader.prepare_train_test()
    return test_df


def _run_evaluation(configs: dict, test_df: pd.DataFrame) -> dict[str, float]:
    """Run reranker evaluation and return metrics."""
    reranker = load_reranker(model_path=configs["model_path"])
    evaluator = ESCIMetricsEvaluator(
        test_df,
        product_col=configs["product_col"],
        max_queries=configs.get("eval_max_queries"),
        batch_size=32,
        recall_at_k=configs["recall_at"],
    )
    evaluator(reranker, output_path=None, epoch=-1, steps=-1)
    return evaluator.last_metrics


if __name__ == "__main__":
    sys.exit(main())
