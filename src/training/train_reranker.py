"""
Train cross-encoder reranker for ESCI Task 1 (query-product ranking).

Training: MSE loss on (query, product) pairs with ESCI gains (E=1.0, S=0.1, C=0.01, I=0.0).
The model predicts a scalar score; we regress toward the gain value.
Uses product_text (expanded) or product_title (ESCI-exact) per config.

Evaluation: nDCG, MRR, MAP, Recall@k on test set. Paper baseline ~0.852 nDCG for US.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


import pandas as pd
import torch
import yaml

from src.constants import DATA_DIR, ESCI_LABEL2GAIN, MODEL_CACHE_DIR, REPO_ROOT, DEFAULT_RERANKER_MODEL
from src.data.load_data import load_esci, prepare_train_test
from src.eval.esci_evaluator import ESCIMetricsEvaluator
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def load_data(base: Path, *, small_version: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test DataFrames via load_esci + prepare_train_test.

    Parameters
    ----------
    base : Path
        Base directory containing ESCI data.
    small_version : bool
        Use Task 1 reduced set (~48k queries) if loading from raw.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Train and test DataFrames.
    """
    df = load_esci(data_dir=base, small_version=small_version)
    return prepare_train_test(df=df)


def build_dataloader(
    train_df: pd.DataFrame,
    *,
    product_col: str,
    batch_size: int,
) -> DataLoader:
    """
    Build DataLoader of InputExamples (query, product) -> gain.

    Parameters
    ----------
    train_df : pd.DataFrame
        Train DataFrame.
    product_col : str
        Column for product text: "product_text" (full) or "product_title" (ESCI exact).
    batch_size : int
        Batch size.

    Returns
    -------
    DataLoader
        DataLoader of InputExamples.
    """
    # Build list of InputExamples (query, product) -> gain
    samples = []

    # Iterate over train DataFrame and build InputExamples
    for _, row in train_df.iterrows():
        gain = ESCI_LABEL2GAIN.get(str(row["esci_label"]), 0.0)

        # query, product -> gain as an InputExample
        samples.append(InputExample(texts=[str(row["query"]), str(row[product_col])], label=float(gain)))

    return DataLoader(samples, shuffle=True, batch_size=batch_size, drop_last=True, pin_memory=False)


def create_model(
    model_name: str,
    *,
    max_length: int = 512,
    device: str | None = None,
    cache_folder: Path | str | None = MODEL_CACHE_DIR,
) -> CrossEncoder:
    """
    Create CrossEncoder with regression head (num_labels=1, Identity activation).
    Uses cache_folder so the model is downloaded once and reused.

    Parameters
    ----------
    model_name : str
        Name of the model to use.
    max_length : int
        Max length of the input sequence.
    device : str | None
        Device to use.
    cache_folder : Path | str | None
        Where to cache downloaded models. Default: data/.model_cache.

    Returns
    -------
    CrossEncoder
        CrossEncoder model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps"
    cache = str(cache_folder) if cache_folder else None
    return CrossEncoder(
        model_name,
        num_labels=1,
        max_length=max_length,
        activation_fn=torch.nn.Identity(),
        device=device,
        cache_folder=cache,
    )


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
    eval_max_queries: int | None = None,
    small_version: bool = False,
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
    eval_max_queries : int | None
        Subsample eval to this many queries (default: all).
    small_version : bool
        Use Task 1 reduced set (~48k queries) if loading from raw.
    """
    # Check if sentence-transformers is installed
    if CrossEncoder is None or InputExample is None:
        raise ImportError("sentence-transformers is required for train_reranker")
    base = Path(data_dir or DATA_DIR)

    # Load train and test data
    train_df, test_df = load_data(base, small_version=small_version)

    logger.info("Data:")
    logger.info("------")
    logger.info("data_dir=%s", base)
    logger.info("small_version=%s", small_version)
    logger.info("product_col=%s", product_col)
    logger.info("train_rows=%d test_rows=%d", len(train_df), len(test_df))
    logger.info("Training:")
    logger.info("------")
    logger.info("model=%s", model_name)
    logger.info("epochs=%d batch_size=%d lr=%g", epochs, batch_size, lr)
    logger.info("warmup_steps=%d max_length=%d", warmup_steps, max_length)
    logger.info("eval_steps=%d eval_max_queries=%s", evaluation_steps, eval_max_queries)
    logger.info("save_path=%s", save_path)

    # Check if train_df has the required columns
    if "esci_label" not in train_df.columns:
        raise ValueError("train_df must have 'esci_label' (from load_esci)")
    if product_col not in train_df.columns:
        raise ValueError(f"train_df must have '{product_col}'")

    # Build train dataloader
    train_dataloader = build_dataloader(train_df, product_col=product_col, batch_size=batch_size)

    # Create model
    model = create_model(model_name, max_length=max_length)
    output_path = str(save_path) if save_path else None

    # Create output directory if it doesn't exist
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Create evaluator if we have test data and evaluation steps are enabled
    evaluator = None
    if len(test_df) > 0 and evaluation_steps > 0:
        evaluator = ESCIMetricsEvaluator(test_df, product_col=product_col, max_queries=eval_max_queries, batch_size=batch_size)

    # Disable pin_memory on MPS (not supported; avoids DataLoader warning)
    if str(model.device) == "mps":
        from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

        _orig_init = CrossEncoderTrainingArguments.__init__

        def _patched_init(self, *args, **kwargs):
            kwargs.setdefault("dataloader_pin_memory", False)
            _orig_init(self, *args, **kwargs)

        CrossEncoderTrainingArguments.__init__ = _patched_init

    model.fit(
        train_dataloader=train_dataloader,
        loss_fct=torch.nn.MSELoss(),
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        optimizer_params={"lr": lr},
        evaluation_steps=evaluation_steps,
        show_progress_bar=True,
    )

    if str(model.device) == "mps":
        CrossEncoderTrainingArguments.__init__ = _orig_init
    if output_path:
        model.save(output_path)
        logger.info("Reranker saved to %s", output_path)
    return model


def main() -> int:
    """CLI entrypoint: parse args, load config, run training."""
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
    p.add_argument("--evaluation-steps", type=int, default=None, help="Eval every N steps (0=disabled)")
    p.add_argument("--eval-max-queries", type=int, default=None, help="Subsample eval queries (default: all)")
    p.add_argument("--small-version", action="store_true", help="Use Task 1 reduced set (~48k queries)")
    args = p.parse_args()
    cfg: dict = {}
    config_path = REPO_ROOT / args.config
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    # CLI overrides config overrides defaults
    run_training(
        data_dir=args.data_dir or cfg.get("data_dir"),
        model_name=args.model_name or cfg.get("model_name", DEFAULT_RERANKER_MODEL),
        product_col=args.product_col or cfg.get("product_col", "product_text"),
        save_path=args.save or cfg.get("save_path", "data/reranker"),
        epochs=args.epochs if args.epochs is not None else cfg.get("epochs", 1),
        batch_size=args.batch_size if args.batch_size is not None else cfg.get("batch_size", 32),
        lr=args.lr if args.lr is not None else cfg.get("lr", 7e-6),
        warmup_steps=args.warmup_steps if args.warmup_steps is not None else cfg.get("warmup_steps", 5000),
        evaluation_steps=args.evaluation_steps if args.evaluation_steps is not None else cfg.get("evaluation_steps", 5000),
        eval_max_queries=args.eval_max_queries or cfg.get("eval_max_queries"),
        small_version=args.small_version or cfg.get("small_version", False),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
