"""
ESCI metrics evaluator: nDCG, MRR, MAP, Recall@k.

Used by train_reranker (mid-training eval) and eval_reranker (standalone eval).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from src.constants import ESCI_LABEL2GAIN

logger = logging.getLogger(__name__)


def compute_query_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    recall_at_k: int = 10,
) -> dict[str, float]:
    """
    Compute nDCG, MRR, MAP, Recall@k for a single query. y_true, y_score: (1, n).

    Parameters
    ----------
    y_true : np.ndarray
        True labels (1, n).
    y_score : np.ndarray
        Predicted scores (1, n).
    recall_at_k : int
        Recall@k to compute.

    Returns
    -------
    dict[str, float]
        Dictionary containing nDCG, MRR, MAP, Recall@k.
    """
    # Rank items by predicted score (descending)
    gains = y_true.flatten()
    scores = y_score.flatten()
    order = np.argsort(-scores)
    ranked_gains = gains[order]

    # Binary relevance: E/S/C = 1, I = 0 (used for MRR, MAP, Recall)
    binary_rel = (gains > 0).astype(np.float64)

    # nDCG: uses graded gains (E=1, S=0.1, C=0.01, I=0); discounts by position
    ndcg = float(ndcg_score(y_true, y_score))

    # MRR: 1 / rank of first relevant item (1-indexed). 0 if none in top-k.
    mrr = 0.0
    for r in range(min(recall_at_k, len(ranked_gains))):
        if ranked_gains[r] > 0:
            mrr = 1.0 / (r + 1)
            break

    # MAP: mean of precision@k for each relevant item when it appears
    n_rel = int(binary_rel.sum())
    if n_rel > 0:
        ranked_binary = binary_rel[order]
        precisions = np.cumsum(ranked_binary) / np.arange(1, len(ranked_binary) + 1)
        ap = (precisions * ranked_binary).sum() / n_rel
    else:
        ap = 0.0

    # Recall@k: fraction of relevant items in top-k
    rel_in_topk = ranked_gains[:recall_at_k] > 0
    recall = rel_in_topk.sum() / max(n_rel, 1)

    return {"ndcg": ndcg, "mrr": mrr, "map": ap, "recall": recall}


class ESCIMetricsEvaluator:
    """
    Evaluator for sentence_transformers CrossEncoder.fit().

    Computes nDCG, MRR, MAP, Recall@k on ESCI test set. Groups by query_id,
    scores (query, product) pairs, ranks by score, computes metrics per query.
    Returns nDCG as primary metric (used by fit for model selection).
    """

    def __init__(
        self,
        test_df: pd.DataFrame,
        product_col: str = "product_text",
        label2gain: dict[str, float] | None = None,
        max_queries: int | None = None,
        batch_size: int = 32,
        recall_at_k: int = 10,
    ):
        self.product_col = product_col
        self.label2gain = label2gain or ESCI_LABEL2GAIN
        self.batch_size = batch_size
        self.recall_at_k = recall_at_k
        # Build list of (query_str, [(product_text, esci_label), ...]) per query_id
        groups = test_df.groupby("query_id")
        self._query_data: list[tuple[str, list[tuple[str, str]]]] = []
        for qid, grp in groups:
            query = str(grp["query"].iloc[0])
            pairs = [(str(row[self.product_col]), str(row["esci_label"])) for _, row in grp.iterrows()]
            self._query_data.append((query, pairs))
        # Optional subsample for faster mid-training eval
        if max_queries and len(self._query_data) > max_queries:
            import random

            rng = random.Random(42)
            self._query_data = rng.sample(self._query_data, max_queries)
        self._last_metrics: dict[str, float] = {}
        logger.info("ESCI metrics evaluator: %d queries", len(self._query_data))

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        """Run evaluation; return nDCG (primary metric for model.fit)."""
        all_metrics: dict[str, list[float]] = {"ndcg": [], "mrr": [], "map": [], "recall": []}
        for query, pairs in self._query_data:
            # Score each (query, product) pair
            texts = [[query, p] for p, _ in pairs]
            y_true = np.array([[self.label2gain.get(lbl, 0.0) for _, lbl in pairs]])
            scores = model.predict(texts, batch_size=self.batch_size, show_progress_bar=False)
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            y_score = np.array([[float(s) for s in scores]])
            m = compute_query_metrics(y_true, y_score, recall_at_k=self.recall_at_k)
            for k, v in m.items():
                all_metrics[k].append(v)
        self._last_metrics = {k: float(np.mean(v)) if v else 0.0 for k, v in all_metrics.items()}
        logger.info(
            "Eval(epoch=%d steps=%d) nDCG=%.4f MRR=%.4f MAP=%.4f Recall@%d=%.4f",
            epoch,
            steps,
            self._last_metrics["ndcg"],
            self._last_metrics["mrr"],
            self._last_metrics["map"],
            self.recall_at_k,
            self._last_metrics["recall"],
        )
        return self._last_metrics["ndcg"]
