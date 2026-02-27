"""
Train cross-encoder reranker using ESCI's approach: MSE loss on (query, product) pairs
with ESCI gains (E=1.0, S=0.1, C=0.01, I=0.0). Uses product_text (or product_title).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.constants import DATA_DIR, REPO_ROOT
from src.data.load_data import load_esci, prepare_train_test

logger = logging.getLogger(__name__)

try:
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.readers import InputExample
    from torch.utils.data import DataLoader
except ImportError:
    CrossEncoder = None
    InputExample = None  # type: ignore[misc, assignment]
    DataLoader = None

ESCI_LABEL2GAIN = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"


def run_training(
    data_dir: Path | str | None = None,
    *,
    model_name: str = DEFAULT_RERANKER_MODEL,
    product_col: str = "product_text",
    save_path: str | Path | None = "data/reranker",
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 7e-6,
    warmup_steps: int = 5000,
    max_length: int = 512,
    evaluation_steps: int = 5000,
    use_saved_splits: bool = False,
):
    """
    Train cross-encoder reranker on ESCI (ESCI baseline approach).

    Parameters
    ----------
    data_dir : Path | str | None
        Directory with ESCI parquets or raw data.
    model_name : str
        Pretrained cross-encoder (e.g. ms-marco-MiniLM-L-12-v2).
    product_col : str
        Column for product text: "product_text" (full) or "product_title" (ESCI exact).
    save_path : str | Path | None
        Where to save the trained model.
    epochs : int
        Number of epochs.
    batch_size : int
        Training batch size.
    lr : float
        Learning rate.
    warmup_steps : int
        Warmup steps.
    max_length : int
        Max sequence length for [query, product].
    evaluation_steps : int
        Evaluate every N steps (0 = no mid-training eval).
    use_saved_splits : bool
        Use esci_train.parquet / esci_test.parquet if present.
    """
    if CrossEncoder is None or InputExample is None:
        raise ImportError("sentence-transformers is required for train_reranker")
    base = Path(data_dir or DATA_DIR)
    train_path = base / "esci_train.parquet"
    if use_saved_splits and train_path.exists():
        train_df = pd.read_parquet(train_path)
    else:
        df = load_esci(data_dir=base)
        train_df, _ = prepare_train_test(df=df)
    if "esci_label" not in train_df.columns:
        raise ValueError("train_df must have 'esci_label' (from load_esci)")
    if product_col not in train_df.columns:
        if product_col == "product_title" and "product_title" in train_df.columns:
            pass
        else:
            raise ValueError(f"train_df must have '{product_col}'")
    train_samples = []
    for _, row in train_df.iterrows():
        gain = ESCI_LABEL2GAIN.get(str(row["esci_label"]), 0.0)
        train_samples.append(
            InputExample(texts=[str(row["query"]), str(row[product_col])], label=float(gain))
        )
    train_dataloader = DataLoader(
        train_samples,
        shuffle=True,
        batch_size=batch_size,
        drop_last=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(
        model_name,
        num_labels=1,
        max_length=max_length,
        activation_fn=torch.nn.Identity(),
        device=device,
    )
    loss_fct = torch.nn.MSELoss()
    output_path = str(save_path) if save_path else None
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_dataloader=train_dataloader,
        loss_fct=loss_fct,
        evaluator=None,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        optimizer_params={"lr": lr},
        evaluation_steps=evaluation_steps,
        show_progress_bar=True,
    )
    if output_path:
        model.save(output_path)
        logger.info("Reranker saved to %s", output_path)
    return model


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Train cross-encoder reranker (ESCI approach)")
    p.add_argument("--config", type=str, default="configs/reranker.yaml")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--product-col", type=str, default=None, choices=["product_text", "product_title"])
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--use-saved-splits", action="store_true")
    args = p.parse_args()
    cfg: dict = {}
    config_path = REPO_ROOT / args.config
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    run_training(
        data_dir=args.data_dir or cfg.get("data_dir"),
        model_name=args.model_name or cfg.get("model_name", DEFAULT_RERANKER_MODEL),
        product_col=args.product_col or cfg.get("product_col", "product_text"),
        save_path=args.save or cfg.get("save_path", "data/reranker"),
        epochs=args.epochs if args.epochs is not None else cfg.get("epochs", 1),
        batch_size=args.batch_size if args.batch_size is not None else cfg.get("batch_size", 32),
        lr=args.lr if args.lr is not None else cfg.get("lr", 7e-6),
        warmup_steps=args.warmup_steps if args.warmup_steps is not None else cfg.get("warmup_steps", 5000),
        use_saved_splits=args.use_saved_splits or cfg.get("use_saved_splits", False),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
