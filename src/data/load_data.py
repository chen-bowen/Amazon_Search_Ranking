"""
Load and prepare Amazon ESCI dataset: merge examples + products, graded relevance, product text expansion.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

from .utils import (
    ESCI_SUBDIR,
    EXAMPLES_FILENAME,
    PRODUCTS_FILENAME,
    esci_label2relevance_pos,
    get_product_expanded_text,
)


# -----------------------------------------------------------------------------
# Loading: read parquets, merge
# -----------------------------------------------------------------------------


def _load_and_merge_parquets(data_dir: Path) -> pd.DataFrame:
    """
    Load examples and products parquets and merge on (product_id, product_locale).

    Parameters
    ----------
    data_dir : Path
        Path to the directory containing the ESCI parquet files.

    Returns
    -------
    pd.DataFrame
        DataFrame with the merged examples and products.
    """
    examples_path = data_dir / EXAMPLES_FILENAME
    products_path = data_dir / PRODUCTS_FILENAME
    df_examples = pd.read_parquet(examples_path)
    df_products = pd.read_parquet(products_path)
    # Left join: keep every example row (query_id, query, product_id, ...) and add product metadata
    return pd.merge(df_examples, df_products, on=["product_id", "product_locale"], how="left")


# -----------------------------------------------------------------------------
# Filtering and relevance
# -----------------------------------------------------------------------------


def _apply_filters(df: pd.DataFrame, *, small_version: bool, locale: str = "us") -> pd.DataFrame:
    """
    Restrict to small_version (Task 1) and a single locale.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the merged examples and products.
    small_version : bool

    Restrict to small_version (Task 1) and a single locale.

    Returns
    -------
    pd.DataFrame
        DataFrame with the filtered examples and products.
    """
    # Filter to small_version (Task 1)
    if small_version:
        df = df[df["small_version"] == 1]

    # Filter to a single locale (US)
    df = df[df["product_locale"] == locale]
    return df


def _add_relevance_column(df: pd.DataFrame, relevance_map: dict[str, int]) -> pd.DataFrame:
    """Map esci_label to numeric relevance; drop rows that don't map."""
    # Add a new column 'relevance' with the numeric relevance score
    df["relevance"] = df["esci_label"].map(relevance_map).astype("int32")

    # Drop rows that don't map to a relevance score
    return df.dropna(subset=["relevance"])


def _add_product_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add column 'product_text' with expanded product string per row."""
    df["product_text"] = df.apply(get_product_expanded_text, axis=1)
    return df


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def load_esci(
    data_dir: Path | str | None = None,
    *,
    small_version: bool = True,
    locale: str = "us",
    relevance_map: dict[str, int] | None = None,
    save_splits_dir: Path | str | None = None,
) -> pd.DataFrame:
    """
    Load ESCI examples + products, merge, filter, and add graded relevance (1-4) and product_text.
    Optionally write train/test parquets to save_splits_dir (esci_train.parquet, esci_test.parquet).

    Parameters
    ----------
    data_dir : Path | str | None
        Directory containing the ESCI parquet files.
    small_version : bool
        Use Task 1 reduced set.
    locale : str
        Locale, e.g. 'us'.
    relevance_map : dict[str, int] | None
        ESCI label -> relevance score (default: E=4, S=2, C=3, I=1).
    save_splits_dir : Path | str | None
        If set, write train/test splits here as esci_train.parquet and esci_test.parquet.

    Returns
    -------
    pd.DataFrame
        Loaded and processed ESCI DataFrame.
    """
    # Resolve the data directory path
    data_dir_path = Path(data_dir or ESCI_SUBDIR)
    examples_path = data_dir_path / EXAMPLES_FILENAME
    products_path = data_dir_path / PRODUCTS_FILENAME
    if not examples_path.exists():
        raise FileNotFoundError(
            f"Examples not found: {examples_path}. "
            "Place ESCI parquet files in data/ or data/esci-data/shopping_queries_dataset/ (see README)."
        )
    if not products_path.exists():
        raise FileNotFoundError(f"Products not found: {products_path}.")

    # Load and merge the examples and products parquets
    df = _load_and_merge_parquets(data_dir_path)

    # apply preprocessing steps
    df = _apply_filters(df, small_version=small_version, locale=locale)
    mapping = relevance_map or esci_label2relevance_pos
    df = _add_relevance_column(df, mapping)
    df = _add_product_text_column(df)

    # If save_splits_dir is set, write the train and test parquets
    if save_splits_dir is not None:
        out_dir = Path(save_splits_dir)
        # Split the data into train and test
        train = df[df["split"] == "train"]
        test = df[df["split"] == "test"]

        # Create the output directory if it doesn't exist, save the train and test parquets
        out_dir.mkdir(parents=True, exist_ok=True)
        train.to_parquet(out_dir / "esci_train.parquet", index=False)
        test.to_parquet(out_dir / "esci_test.parquet", index=False)
        logger.info("Train: %s rows -> %s", len(train), out_dir / "esci_train.parquet")
        logger.info("Test: %s rows -> %s", len(test), out_dir / "esci_test.parquet")

    return df


if __name__ == "__main__":
    import argparse
    import sys
    from .utils import DATA_DIR

    # Configure logging to only show messages
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Parse command-line arguments
    p = argparse.ArgumentParser(description="Load ESCI data; optionally save train/test parquets")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--save-splits", action="store_true", help="Write esci_train.parquet and esci_test.parquet to data/")
    args = p.parse_args()

    # Resolve the data directory path
    data_dir = Path(args.data_dir) if args.data_dir else None
    if data_dir is None and (DATA_DIR / EXAMPLES_FILENAME).exists():
        data_dir = DATA_DIR

    # Load the ESCI data
    load_esci(data_dir=data_dir, save_splits_dir=DATA_DIR if args.save_splits else None)
    sys.exit(0)
