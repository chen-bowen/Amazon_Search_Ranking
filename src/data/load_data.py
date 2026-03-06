"""
Load and prepare Amazon ESCI dataset: merge examples + products, graded
relevance, product text expansion.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import (
    ESCI_SUBDIR,
    EXAMPLES_FILENAME,
    PRODUCTS_FILENAME,
    esci_label2relevance_pos,
    get_product_expanded_text,
)

logger = logging.getLogger(__name__)


class ESCIDataLoader:
    """
    ESCI data loader: load parquets, merge, filter, enrich, and prepare splits.

    Implements all data loading logic. Configure via constructor; call load_esci(),
    prepare_train_test(), or prepare_train_val_test().
    """

    def __init__(
        self,
        data_dir: Path | str | None = None,
        *,
        small_version: bool = False,
        locale: str = "us",
        relevance_map: dict[str, int] | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.small_version = small_version
        self.locale = locale
        self.relevance_map = relevance_map

    def load_esci(
        self,
        *,
        save_splits_dir: Path | str | None = None,
    ) -> pd.DataFrame:
        """
        Load, merge, filter, and enrich ESCI data according to this loader's configuration.
        """
        data_dir_path = self._resolve_data_dir()
        self._validate_data_paths(data_dir_path)

        df = self._load_and_merge_parquets(data_dir_path)
        df = self._apply_filters(df)
        df = self._add_relevance_column(df)
        df = self._add_product_text_column(df)

        if save_splits_dir is not None:
            self._save_train_test_splits(df, Path(save_splits_dir))

        return df

    def prepare_train_test(
        self,
        df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return (train_df, test_df) by splitting on the "split" column.
        If df is None, load via load_esci() first.
        """
        if df is None:
            df = self.load_esci()
        if "split" not in df.columns:
            raise ValueError('DataFrame has no "split" column; cannot split train/test.')
        train = df[df["split"] == "train"].copy()
        test = df[df["split"] == "test"].copy()
        return train, test

    def prepare_train_val_test(
        self,
        df: pd.DataFrame | None = None,
        *,
        val_frac: float = 0.1,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Return (train_df, val_df, test_df). Train/test from "split" column;
        val is a held-out subset of train by query_id.
        """
        train, test = self.prepare_train_test(df=df)
        if val_frac <= 0 or val_frac >= 1 or "query_id" not in train.columns:
            return train, pd.DataFrame(), test

        query_ids = train["query_id"].unique()
        _, val_qids = train_test_split(query_ids, test_size=val_frac, random_state=random_state)
        val_ids = set(val_qids)
        val = train[train["query_id"].isin(val_ids)].copy()
        train = train[~train["query_id"].isin(val_ids)].copy()
        return train, val, test

    def _resolve_data_dir(self) -> Path:
        """Resolve the base data directory."""
        return Path(self.data_dir or ESCI_SUBDIR)

    def _load_and_merge_parquets(self, data_dir: Path) -> pd.DataFrame:
        """Load examples and products parquets and merge on id + locale."""
        examples_path = data_dir / EXAMPLES_FILENAME
        products_path = data_dir / PRODUCTS_FILENAME
        df_examples = pd.read_parquet(examples_path)
        df_products = pd.read_parquet(products_path)
        return pd.merge(
            df_examples,
            df_products,
            on=["product_id", "product_locale"],
            how="left",
        )

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply small_version and locale filters."""
        if self.small_version:
            df = df[df["small_version"] == 1]
        df = df[df["product_locale"] == self.locale]
        return df

    def _add_relevance_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach numeric relevance column from esci_label."""
        mapping = self.relevance_map or esci_label2relevance_pos
        df["relevance"] = df["esci_label"].map(mapping).astype("int32")
        return df.dropna(subset=["relevance"])

    def _add_product_text_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach expanded product_text column."""
        df["product_text"] = df.apply(get_product_expanded_text, axis=1)
        return df

    def _validate_data_paths(self, data_dir: Path) -> None:
        """Ensure ESCI parquet files exist in the given directory."""
        examples_path = data_dir / EXAMPLES_FILENAME
        products_path = data_dir / PRODUCTS_FILENAME

        if not examples_path.exists():
            raise FileNotFoundError(f"Examples not found: {examples_path}. " "Place ESCI parquet files in data/ (see README).")
        if not products_path.exists():
            raise FileNotFoundError(f"Products not found: {products_path}.")

    def _save_train_test_splits(self, df: pd.DataFrame, out_dir: Path) -> None:
        """Persist train and test splits to parquet files in out_dir."""
        train = df[df["split"] == "train"]
        test = df[df["split"] == "test"]

        out_dir.mkdir(parents=True, exist_ok=True)
        train_path = out_dir / "esci_train.parquet"
        test_path = out_dir / "esci_test.parquet"

        train.to_parquet(train_path, index=False)
        test.to_parquet(test_path, index=False)

        logger.info("Train: %s rows -> %s", len(train), train_path)
        logger.info("Test: %s rows -> %s", len(test), test_path)


if __name__ == "__main__":
    import argparse
    import sys
    from .utils import DATA_DIR

    # Configure logging to only show messages
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Parse command-line arguments
    p = argparse.ArgumentParser(description="Load ESCI data; optionally save train/test parquets")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument(
        "--save-splits",
        action="store_true",
        help="Write esci_train.parquet and esci_test.parquet to data/",
    )
    args = p.parse_args()

    # Resolve the data directory path
    data_dir = Path(args.data_dir) if args.data_dir else None
    if data_dir is None and (DATA_DIR / EXAMPLES_FILENAME).exists():
        data_dir = DATA_DIR

    # Load the ESCI data
    loader = ESCIDataLoader(data_dir=data_dir)
    loader.load_esci(save_splits_dir=DATA_DIR if args.save_splits else None)
    sys.exit(0)
