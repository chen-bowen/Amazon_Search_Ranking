"""
ESCI metrics evaluator: nDCG, MRR, MAP, Recall@k.

Used by train_reranker (mid-training eval) and eval_reranker (standalone eval).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, ndcg_score
from tqdm import tqdm

from src.constants import ESCI_LABEL2GAIN, ESCI_LABEL2ID
from src.utils import clear_torch_cache

logger = logging.getLogger(__name__)


def compute_query_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    recall_at_k: int = 10,
) -> dict[str, float]:
    """
    Compute nDCG, MRR, MAP, Recall@k for a single query.
    y_true, y_score: (1, n).

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
    gains = y_true.flatten()
    scores = y_score.flatten()
    order = np.argsort(-scores)
    ranked_gains = gains[order]

    # Binary relevance: E/S/C = 1, I = 0 (used for MRR, MAP, Recall)
    binary_rel = (gains > 0).astype(np.float64)

    # nDCG: uses graded gains (E=1, S=0.1, C=0.01, I=0); discounts by position
    ndcg = float(ndcg_score(y_true, y_score))
    n_rel = int(binary_rel.sum())

    mrr = _compute_mrr(ranked_gains, recall_at_k)
    ap = _compute_map(binary_rel, order, n_rel)
    recall = _compute_recall(ranked_gains, recall_at_k, n_rel)

    return {"ndcg": ndcg, "mrr": mrr, "map": ap, "recall": recall}


def _compute_mrr(ranked_gains: np.ndarray, recall_at_k: int) -> float:
    """MRR: 1 / rank of first relevant item (1-indexed). 0 if none in top-k."""
    mrr = 0.0
    limit = min(recall_at_k, len(ranked_gains))
    for r in range(limit):
        if ranked_gains[r] > 0:
            mrr = 1.0 / (r + 1)
            break
    return mrr


def _compute_map(
    binary_rel: np.ndarray,
    order: np.ndarray,
    n_rel: int,
) -> float:
    """MAP: mean of precision@k for each relevant item when it appears."""
    if n_rel <= 0:
        return 0.0
    ranked_binary = binary_rel[order]
    precisions = np.cumsum(ranked_binary) / np.arange(1, len(ranked_binary) + 1)
    return float((precisions * ranked_binary).sum() / n_rel)


def _compute_recall(
    ranked_gains: np.ndarray,
    recall_at_k: int,
    n_rel: int,
) -> float:
    """Recall@k: fraction of relevant items in top-k."""
    if n_rel <= 0:
        return 0.0
    rel_in_topk = ranked_gains[:recall_at_k] > 0
    return float(rel_in_topk.sum() / n_rel)


class ESCIMetricsEvaluator:
    """
    Evaluator for sentence_transformers CrossEncoder.fit().

    Computes nDCG, MRR, MAP, Recall@k on ESCI test set. Groups by query_id,
    scores (query, product) pairs, ranks by score, computes metrics per
    query. Returns nDCG as primary metric (used by fit for model selection).
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
            pairs = [
                (str(row[self.product_col]), str(row["esci_label"]))
                for _, row in grp.iterrows()
            ]
            self._query_data.append((query, pairs))
        # Optional subsample for faster mid-training eval
        if max_queries and len(self._query_data) > max_queries:
            import random

            rng = random.Random(42)
            self._query_data = rng.sample(self._query_data, max_queries)
        self._last_metrics: dict[str, float] = {}
        logger.info("ESCI metrics evaluator: %d queries", len(self._query_data))

    @property
    def last_metrics(self) -> dict[str, float]:
        """Last computed metrics (ndcg, mrr, map, recall) after __call__."""
        return self._last_metrics

    def __call__(
        self, model, output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> float:
        """Run evaluation; return nDCG (primary metric for model.fit)."""
        clear_torch_cache()
        all_metrics: dict[str, list[float]] = {
            "ndcg": [],
            "mrr": [],
            "map": [],
            "recall": [],
        }
        for query, pairs in tqdm(self._query_data, desc="Eval", unit="query"):
            m = self._score_query(model, query, pairs)
            for k, v in m.items():
                all_metrics[k].append(v)
        return self._aggregate_and_log(all_metrics, epoch=epoch, steps=steps)

    def _score_query(
        self,
        model,
        query: str,
        pairs: list[tuple[str, str]],
    ) -> dict[str, float]:
        """Score a single query's pairs and compute metrics."""
        texts = [[query, p] for p, _ in pairs]
        y_true = np.array([[self.label2gain.get(lbl, 0.0) for _, lbl in pairs]])
        scores = model.predict(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        y_score = np.array([[float(s) for s in scores]])
        return compute_query_metrics(y_true, y_score, recall_at_k=self.recall_at_k)

    def _aggregate_and_log(
        self,
        all_metrics: dict[str, list[float]],
        *,
        epoch: int,
        steps: int,
    ) -> float:
        """Aggregate per-query metrics, log, and return nDCG."""
        self._last_metrics = {
            k: float(np.mean(v)) if v else 0.0 for k, v in all_metrics.items()
        }
        msg = (
            f"Eval(epoch={epoch} steps={steps}) "
            f"nDCG={self._last_metrics['ndcg']:.4f} "
            f"MRR={self._last_metrics['mrr']:.4f} "
            f"MAP={self._last_metrics['map']:.4f} "
            f"Recall@{self.recall_at_k}={self._last_metrics['recall']:.4f}"
        )
        # Leading newline so eval metrics appear on a new line, not after progress bar
        tqdm.write("\n" + msg)
        return self._last_metrics["ndcg"]


def evaluate_classification_tasks(
    model,
    df: pd.DataFrame,
    *,
    product_col: str,
    max_queries: int | None = None,
    batch_size: int = 32,
    split_name: str = "val",
) -> None:
    """
    Evaluate ESCI multi-task classification heads on a dataframe.

    Metrics:
    - Task 2 (4-class ESCI): accuracy and macro F1 over E/S/C/I.
    - Task 3 (substitute): accuracy and F1 for substitute vs non-substitute.

    The model is expected to expose `predict(pairs)` returning
    (scores, esci_labels, substitute_probs).
    """
    evaluator = ClassificationTaskEvaluator(
        df=df,
        product_col=product_col,
        max_queries=max_queries,
        batch_size=batch_size,
        split_name=split_name,
    )
    evaluator(model)


class ClassificationTaskEvaluator:
    """
    Helper to evaluate ESCI multi-task classification heads on a dataframe.

    Computes:
    - Task 2 (4-class ESCI): accuracy and macro F1 over E/S/C/I.
    - Task 3 (substitute): accuracy and F1 for substitute vs non-substitute.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        product_col: str,
        max_queries: int | None,
        batch_size: int,
        split_name: str,
    ) -> None:
        self._df = df
        self._product_col = product_col
        self._max_queries = max_queries
        self._batch_size = batch_size
        self._split_name = split_name

    def __call__(self, model) -> None:
        if self._df.empty:
            return

        df_eval = self._prepare_eval_data()
        pairs, true_esci_ids, true_sub = self._build_inputs(df_eval)
        pred_esci_ids, pred_sub = self._predict(model, pairs)
        self._compute_and_log_metrics(
            true_esci_ids=true_esci_ids,
            pred_esci_ids=pred_esci_ids,
            true_sub=true_sub,
            pred_sub=pred_sub,
            n_examples=len(df_eval),
        )

    def _prepare_eval_data(self) -> pd.DataFrame:
        if self._max_queries is not None and "query_id" in self._df.columns:
            qids = self._df["query_id"].unique()[: self._max_queries]
            return self._df[self._df["query_id"].isin(qids)].copy()
        return self._df

    def _build_inputs(
        self,
        df_eval: pd.DataFrame,
    ) -> tuple[list[list[str]], list[int], list[int]]:
        pairs = [
            [str(r["query"]), str(r[self._product_col])]
            for _, r in df_eval.iterrows()
        ]
        true_labels = [str(r["esci_label"]) for _, r in df_eval.iterrows()]
        true_esci_ids = [ESCI_LABEL2ID[lbl] for lbl in true_labels]
        true_sub = [1 if lbl == "S" else 0 for lbl in true_labels]
        return pairs, true_esci_ids, true_sub

    def _predict(
        self,
        model,
        pairs: list[list[str]],
    ) -> tuple[list[int], list[int]]:
        _, pred_esci_labels, sub_probs = model.predict(
            pairs,
            batch_size=self._batch_size,
        )
        pred_esci_ids = [ESCI_LABEL2ID[lbl] for lbl in pred_esci_labels]
        pred_sub = [1 if p >= 0.5 else 0 for p in sub_probs]
        return pred_esci_ids, pred_sub

    def _compute_and_log_metrics(
        self,
        *,
        true_esci_ids: list[int],
        pred_esci_ids: list[int],
        true_sub: list[int],
        pred_sub: list[int],
        n_examples: int,
    ) -> None:
        esci_acc = accuracy_score(true_esci_ids, pred_esci_ids)
        esci_f1 = f1_score(true_esci_ids, pred_esci_ids, average="macro")
        sub_acc = accuracy_score(true_sub, pred_sub)
        sub_f1 = f1_score(true_sub, pred_sub)

        logger.info(
            "[%s] Task 2 (ESCI 4-class): accuracy=%.4f macro_F1=%.4f on %d examples",
            self._split_name,
            esci_acc,
            esci_f1,
            n_examples,
        )
        logger.info(
            "[%s] Task 3 (substitute): accuracy=%.4f F1=%.4f on %d examples",
            self._split_name,
            sub_acc,
            sub_f1,
            n_examples,
        )
