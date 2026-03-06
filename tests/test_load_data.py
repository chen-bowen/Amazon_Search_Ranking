"""Tests for src.data.load_data."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.load_data import ESCIDataLoader


def test_prepare_train_test_from_df() -> None:
    """ESCIDataLoader.prepare_train_test splits on 'split' column."""
    df = pd.DataFrame(
        {
            "query_id": [1, 2, 3, 4],
            "query": ["a", "b", "c", "d"],
            "split": ["train", "train", "test", "test"],
        }
    )
    loader = ESCIDataLoader()
    train, test = loader.prepare_train_test(df=df)
    assert len(train) == 2
    assert len(test) == 2
    assert list(train["split"].unique()) == ["train"]
    assert list(test["split"].unique()) == ["test"]


def test_prepare_train_test_no_split_column() -> None:
    """Raises ValueError when 'split' column is missing."""
    df = pd.DataFrame({"query_id": [1], "query": ["a"]})
    loader = ESCIDataLoader()
    with pytest.raises(ValueError, match='no "split" column'):
        loader.prepare_train_test(df=df)


def test_prepare_train_test_empty_splits() -> None:
    """Handles empty train or test."""
    df = pd.DataFrame(
        {
            "query_id": [1],
            "query": ["a"],
            "split": ["test"],
        }
    )
    loader = ESCIDataLoader()
    train, test = loader.prepare_train_test(df=df)
    assert len(train) == 0
    assert len(test) == 1


def test_prepare_train_val_test() -> None:
    """ESCIDataLoader.prepare_train_val_test splits train by query_id; test is held out."""
    df = pd.DataFrame(
        {
            "query_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "query": ["a"] * 10,
            "split": ["train"] * 8 + ["test"] * 2,
        }
    )
    loader = ESCIDataLoader()
    train, val, test = loader.prepare_train_val_test(df=df, val_frac=0.2, random_state=42)
    assert len(test) == 2
    assert len(train) + len(val) == 8
    train_qids = set(train["query_id"].unique())
    val_qids = set(val["query_id"].unique())
    assert train_qids.isdisjoint(val_qids)
