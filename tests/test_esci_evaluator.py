"""Tests for src.eval.esci_evaluator."""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.esci_evaluator import compute_query_metrics


def test_compute_query_metrics_perfect_ranking() -> None:
    """When scores match true order, nDCG=1, MRR=1, MAP=1, Recall=1."""
    y_true = np.array([[1.0, 0.1, 0.01, 0.0]])  # E, S, C, I
    y_score = np.array([[1.0, 0.5, 0.2, 0.0]])  # same order
    m = compute_query_metrics(y_true, y_score, recall_at_k=10)
    assert m["ndcg"] == pytest.approx(1.0, abs=1e-5)
    assert m["mrr"] == 1.0
    assert m["map"] == pytest.approx(1.0, abs=1e-5)
    assert m["recall"] == 1.0


def test_compute_query_metrics_worst_ranking() -> None:
    """When relevant items are ranked last, metrics are low."""
    y_true = np.array([[1.0, 0.0, 0.0, 0.0]])  # one relevant at index 0
    y_score = np.array([[0.1, 0.2, 0.3, 0.4]])  # relevant ranked last
    m = compute_query_metrics(y_true, y_score, recall_at_k=10)
    assert m["ndcg"] < 0.6  # nDCG penalizes bad ranking
    assert m["mrr"] == 0.25  # 1/4
    assert m["recall"] == 1.0  # still in top-10 (only 4 items)


def test_compute_query_metrics_no_relevant() -> None:
    """When no relevant items, MRR=0, MAP=0, Recall=0."""
    y_true = np.array([[0.0, 0.0, 0.0]])
    y_score = np.array([[1.0, 0.5, 0.0]])
    m = compute_query_metrics(y_true, y_score, recall_at_k=10)
    assert m["mrr"] == 0.0
    assert m["map"] == 0.0
    assert m["recall"] == 0.0


def test_compute_query_metrics_first_relevant_at_2() -> None:
    """MRR = 1/rank of first relevant (1-indexed)."""
    y_true = np.array([[0.0, 1.0, 0.0, 0.0]])  # relevant at index 1
    y_score = np.array([[1.0, 0.5, 0.3, 0.1]])  # rank: 0, 1, 2, 3 -> relevant at pos 2
    m = compute_query_metrics(y_true, y_score, recall_at_k=10)
    assert m["mrr"] == 0.5  # 1/2


def test_compute_query_metrics_recall_at_k() -> None:
    """Recall@k counts relevant in top-k."""
    y_true = np.array([[1.0, 1.0, 1.0, 0.0]])  # 3 relevant
    y_score = np.array([[1.0, 0.9, 0.1, 0.8]])  # order: 0, 1, 3, 2
    # Top-2 by score: items 0, 1 -> 2 relevant of 3 -> recall = 2/3
    m = compute_query_metrics(y_true, y_score, recall_at_k=2)
    assert m["recall"] == pytest.approx(2 / 3, abs=1e-5)
