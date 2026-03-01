"""Tests for src.data.utils."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.utils import get_product_expanded_text


def test_get_product_expanded_text_full_row() -> None:
    """Product text includes title, brand, bullets, description, color."""
    row = pd.Series({
        "product_title": "Wireless Mouse",
        "product_brand": "Logitech",
        "product_bullet_point": "Ergonomic design",
        "product_description": "A great mouse.",
        "product_color": "Black",
    })
    text = get_product_expanded_text(row)
    assert "[PN]" in text and "Wireless Mouse" in text
    assert "[PBN]" in text and "Logitech" in text
    assert "[PBP]" in text and "Ergonomic design" in text
    assert "[PD]" in text and "A great mouse" in text
    assert "[PCL]" in text and "Black" in text


def test_get_product_expanded_text_title_only() -> None:
    """Missing fields are omitted; fallback when only title exists."""
    row = pd.Series({"product_title": "Basic Product"})
    text = get_product_expanded_text(row)
    assert "[PN]" in text and "Basic Product" in text


def test_get_product_expanded_text_empty_row() -> None:
    """All missing yields fallback."""
    row = pd.Series({})
    text = get_product_expanded_text(row)
    assert "[PN]" in text
    assert "(no title)" in text or "no title" in text.lower()


def test_get_product_expanded_text_nan_handling() -> None:
    """NaN values are treated as empty."""
    row = pd.Series({
        "product_title": "Title",
        "product_brand": pd.NA,
        "product_bullet_point": None,
        "product_description": float("nan"),
        "product_color": "",
    })
    text = get_product_expanded_text(row)
    assert "[PN]" in text and "Title" in text
