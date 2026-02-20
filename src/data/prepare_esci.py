"""
Prepare ESCI train/test splits and save to data/ for fast loading during training.
"""
from __future__ import annotations

import sys
from pathlib import Path

from .load_esci import load_esci, prepare_train_test

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


def main() -> int:
    df = load_esci(data_dir=DATA_DIR / "esci-data" / "shopping_queries_dataset", small_version=True, locale="us")
    train, test = prepare_train_test(df=df)
    train_path = DATA_DIR / "esci_train.parquet"
    test_path = DATA_DIR / "esci_test.parquet"
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    print(f"Train: {len(train)} rows -> {train_path}")
    print(f"Test: {len(test)} rows -> {test_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
