"""Tests for src.data.load_data."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.load_data import prepare_train_test


def test_prepare_train_test_from_df() -> None:
    """prepare_train_test splits on 'split' column."""
    df = pd.DataFrame({
        "query_id": [1, 2, 3, 4],
        "query": ["a", "b", "c", "d"],
        "split": ["train", "train", "test", "test"],
    })
    train, test = prepare_train_test(df=df)
    assert len(train) == 2
    assert len(test) == 2
    assert list(train["split"].unique()) == ["train"]
    assert list(test["split"].unique()) == ["test"]


def test_prepare_train_test_no_split_column() -> None:
    """Raises ValueError when 'split' column is missing."""
    df = pd.DataFrame({"query_id": [1], "query": ["a"]})
    with pytest.raises(ValueError, match='no "split" column'):
        prepare_train_test(df=df)


def test_prepare_train_test_empty_splits() -> None:
    """Handles empty train or test."""
    df = pd.DataFrame({
        "query_id": [1],
        "query": ["a"],
        "split": ["test"],
    })
    train, test = prepare_train_test(df=df)
    assert len(train) == 0
    assert len(test) == 1
