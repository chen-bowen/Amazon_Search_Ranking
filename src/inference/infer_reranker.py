#!/usr/bin/env python3
"""
Run inference with the trained ESCI reranker on a single query.

Loads a CrossEncoder reranker from disk (matching the training config) and
logs the top-k products for a query from the test set.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from src.constants import INFER_RERANKER_DEFAULTS, REPO_ROOT
from src.data.load_data import ESCIDataLoader
from src.models.reranker import load_reranker
from src.utils import load_config

logger = logging.getLogger(__name__)


class RerankerInference:
    """
    Run reranker inference on a single query from the test set.

    Configure via constructor; call run() to execute. Returns 0 on success, 1 on error.
    """

    def __init__(self, configs: dict) -> None:
        self.model_path = configs["model_path"]
        self.data_dir = Path(configs["data_dir"])
        self.product_col = configs.get("product_col", "product_text")
        self.query_override = configs.get("query")
        self.small_version = bool(configs.get("small_version", False))
        self.batch_size = int(configs.get("batch_size", 16))
        self.top_k = int(configs.get("top_k", 5))
        self.query_index = int(configs.get("query_index", 0))

    def run(self) -> int:
        """Run inference; return 0 on success, 1 on error."""
        test_df = self._load_test_df()
        if test_df.empty:
            logger.error("No test data found.")
            return 1

        query, qid = self._select_query(test_df)
        if self.query_override is not None:
            query = str(self.query_override)
        logger.info("Using query_index=%d (query_id=%s)", self.query_index, qid)
        logger.info("Query: %s", query)

        rows = test_df[test_df["query_id"] == qid]
        if len(rows) == 0:
            logger.error("No products found for query_id %s", qid)
            return 1

        candidates = self._prepare_candidates(rows)
        if not candidates:
            return 1

        reranker = load_reranker(model_path=self.model_path)
        ranked = reranker.rerank(query, candidates, batch_size=self.batch_size)
        self._log_ranked_results(ranked, candidates, rows, qid)
        return 0

    def _load_test_df(self) -> pd.DataFrame:
        """Load ESCI test split, preferring pre-saved parquet if available."""
        test_path = self.data_dir / "esci_test.parquet"
        if test_path.exists():
            return pd.read_parquet(test_path)
        loader = ESCIDataLoader(data_dir=self.data_dir, small_version=self.small_version)
        _, test_df = loader.prepare_train_test()
        return test_df

    def _select_query(self, test_df: pd.DataFrame) -> tuple[str, int]:
        """Select query and query_id from test set by index."""
        by_qid = test_df.groupby("query_id").first().reset_index()
        if len(by_qid) == 0:
            raise ValueError("No queries found in test set.")
        if self.query_index < 0 or self.query_index >= len(by_qid):
            raise IndexError(
                f"query_index {self.query_index} out of range 0..{len(by_qid) - 1}"
            )
        row = by_qid.iloc[self.query_index]
        return str(row["query"]), int(row["query_id"])

    def _prepare_candidates(self, rows: pd.DataFrame) -> list[tuple[str, str]]:
        """Build candidate (product_id, product_text) tuples; validate column."""
        if self.product_col not in rows.columns:
            logger.error(
                "Column '%s' not in test data; available: %s",
                self.product_col,
                list(rows.columns),
            )
            return []
        return [
            (str(r["product_id"]), str(r[self.product_col]))
            for _, r in rows.iterrows()
        ]

    def _log_ranked_results(
        self,
        ranked: list[tuple[str, float]],
        candidates: list[tuple[str, str]],
        rows: pd.DataFrame,
        qid: int,
    ) -> None:
        """Log top-k ranked products with labels and truncated text."""
        labels = rows.get("esci_label", ["?"] * len(rows))
        pid_to_label = dict(zip(rows["product_id"].astype(str), labels))

        logger.info("Top %d products for query (query_id=%s):", self.top_k, qid)
        for rank, (pid, score) in enumerate(ranked[: self.top_k], start=1):
            label = pid_to_label.get(pid, "?")
            text = next(t for p, t in candidates if p == pid)
            logger.info("%d. [label=%s] product_id=%s score=%.4f", rank, label, pid, score)
            logger.info("    %s...", text[:200])


def main() -> int:
    """CLI entrypoint: run inference with the trained reranker on one query."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    p = argparse.ArgumentParser(
        description="Run reranker inference on a sample test query."
    )
    p.add_argument(
        "--config", default="configs/reranker.yaml", help="Path to YAML config."
    )
    p.add_argument(
        "--query",
        type=str,
        default=None,
        help="Override query text directly (candidates still come from selected query_id).",
    )
    p.add_argument(
        "--query-index",
        type=int,
        default=None,
        help="Override query_index from config (index over unique query_id values).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override top_k from config (number of results to log).",
    )
    args = p.parse_args()

    config_path = REPO_ROOT / args.config
    cfg = load_config(config_path, INFER_RERANKER_DEFAULTS)

    if args.query_index is not None:
        cfg = cfg or {}
        cfg["query_index"] = args.query_index
    if args.top_k is not None:
        cfg = cfg or {}
        cfg["top_k"] = args.top_k
    if args.query is not None:
        cfg = cfg or {}
        cfg["query"] = args.query

    return RerankerInference(cfg or {}).run()


if __name__ == "__main__":
    sys.exit(main())
