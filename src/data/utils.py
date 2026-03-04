"""
Constants and helpers for ESCI data loading and product text expansion.
"""

from __future__ import annotations


import pandas as pd
from src.constants import DATA_DIR

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# ESCI label -> numeric relevance (higher = more relevant). Used for ranking and metrics.
esci_label2relevance_pos = {
    "E": 4,  # Exact match
    "S": 2,  # Substitute
    "C": 3,  # Complement
    "I": 1,  # Irrelevant
}

# Project paths (from repo root)
ESCI_SUBDIR = DATA_DIR

# Max length per field when building product text (keeps input within model limits)
MAX_TITLE_LEN = 200
MAX_DESC_LEN = 256
MAX_BULLET_LEN = 300

# Parquet filenames expected in data_dir
EXAMPLES_FILENAME = "shopping_queries_dataset_examples.parquet"
PRODUCTS_FILENAME = "shopping_queries_dataset_products.parquet"


# -----------------------------------------------------------------------------
# Helpers: safe string and product text building
# -----------------------------------------------------------------------------


def _safe_str(value: object, max_len: int | None = None) -> str:
    """Turn a value into a string; treat None/NaN as empty; optionally truncate at word boundary."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    if max_len is not None and len(s) > max_len:
        s = s[:max_len].rsplit(" ", 1)[0] if " " in s[:max_len] else s[:max_len]
    return s


def _format_product_part(prefix: str, value: object, max_len: int | None = None) -> str:
    """One segment of product text, e.g. '[PN] title' or '[PBN] brand'. Empty if value is missing."""
    text = _safe_str(value, max_len)
    return f"{prefix} {text}" if text else ""


def get_product_expanded_text(row: pd.Series) -> str:
    """
    Build a single product text string with special tokens (Instacart-style).
    Order: [PN] title, [PBN] brand, [PBP] bullets, [PD] description, [PCL] color.
    """
    parts = [
        _format_product_part("[PN]", row.get("product_title"), MAX_TITLE_LEN),
        _format_product_part("[PBN]", row.get("product_brand")),
        _format_product_part("[PBP]", row.get("product_bullet_point"), MAX_BULLET_LEN),
        _format_product_part("[PD]", row.get("product_description"), MAX_DESC_LEN),
        _format_product_part("[PCL]", row.get("product_color")),
    ]
    parts = [p for p in parts if p]
    return " ".join(parts) if parts else "[PN] (no title)"
