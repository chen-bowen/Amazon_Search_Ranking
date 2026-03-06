"""
Train cross-encoder reranker for ESCI Task 1 (query-product ranking).

Training: MSE loss on (query, product) pairs with ESCI gains
(E=1.0, S=0.1, C=0.01, I=0.0). The model predicts a scalar score; we regress
toward the gain value. Uses product_text (expanded) or product_title
(ESCI-exact) per config.

Evaluation: nDCG, MRR, MAP, Recall@k on test set. Paper baseline ~0.852 nDCG
for US.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

from src.constants import (
    DATA_DIR,
    ESCI_LABEL2GAIN,
    MODEL_CACHE_DIR,
    REPO_ROOT,
    DEFAULT_RERANKER_MODEL,
)
from src.utils import resolve_device, load_config
from src.data.load_data import ESCIDataLoader
from src.eval.evaluator import ESCIMetricsEvaluator
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.evaluation import SequentialEvaluator
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# Defaults for config + CLI (config overrides these; CLI overrides config)
DEFAULTS = {
    "data_dir": "data",
    "model_name": DEFAULT_RERANKER_MODEL,
    "product_col": "product_text",
    "save_path": "checkpoints/reranker",
    "epochs": 1,
    "batch_size": 16,
    "max_length": 512,
    "lr": 7e-6,
    "warmup_steps": 5000,
    "evaluation_steps": 15000,
    "early_stopping_patience": 0,
    "val_frac": 0.1,
}


def build_dataloader(
    train_df: pd.DataFrame,
    *,
    product_col: str,
    batch_size: int,
) -> DataLoader:
    """Build DataLoader of InputExamples for CrossEncoder.fit()."""
    samples = []
    for _, row in train_df.iterrows():
        gain = ESCI_LABEL2GAIN.get(str(row["esci_label"]), 0.0)
        samples.append(
            InputExample(
                texts=[str(row["query"]), str(row[product_col])], label=float(gain)
            )
        )
    return DataLoader(samples, shuffle=True, batch_size=batch_size, drop_last=True)


def create_model(
    model_name: str,
    *,
    max_length: int = 512,
    device: str | None = None,
    cache_folder: Path | str | None = MODEL_CACHE_DIR,
) -> CrossEncoder:
    """
    Create CrossEncoder with regression head (num_labels=1, Identity
    activation). Uses cache_folder so the model is downloaded once and reused.

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
    device = str(resolve_device(device))
    cache = str(cache_folder) if cache_folder else None
    return CrossEncoder(
        model_name,
        num_labels=1,
        max_length=max_length,
        activation_fn=torch.nn.Identity(),
        device=device,
        cache_folder=cache,
    )


class RerankerTrainer:
    """Orchestrates data loading, training, and evaluation for the reranker."""

    def __init__(
        self,
        *,
        data_dir: Path | str | None,
        model_name: str,
        product_col: str,
        save_path: str | Path | None,
        epochs: int,
        batch_size: int,
        lr: float,
        warmup_steps: int,
        max_length: int,
        evaluation_steps: int,
        eval_max_queries: int | None,
        small_version: bool,
        device: str | None,
        early_stopping_patience: int,
        val_frac: float,
    ) -> None:
        self.data_dir = data_dir
        self.model_name = model_name
        self.product_col = product_col
        self.save_path = save_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.max_length = max_length
        self.evaluation_steps = evaluation_steps
        self.eval_max_queries = eval_max_queries
        self.small_version = small_version
        self.device = device
        self.early_stopping_patience = early_stopping_patience
        self.val_frac = val_frac

        self.base = Path(self.data_dir or DATA_DIR)
        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None
        self.model: CrossEncoder | None = None
        self.output_path = str(self.save_path) if self.save_path else None

    def run(self) -> CrossEncoder:
        self._load_splits()
        self._maybe_select_device()
        self._log_data_config()
        self._validate_train_columns()
        self._setup_model()
        train_dataloader = self._build_train_dataloader()
        evaluator = self._build_val_evaluator()
        self._fit_model(train_dataloader, evaluator)
        self._save_model()
        self._run_final_eval()
        return self.model  # type: ignore[return-value]

    def _load_splits(self) -> None:
        loader = ESCIDataLoader(data_dir=self.base, small_version=self.small_version)
        train_df, val_df, test_df = loader.prepare_train_val_test(val_frac=self.val_frac)
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

    def _maybe_select_device(self) -> None:
        if self.device is None and torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("Using MPS (Apple Silicon GPU) for training.")

    def _log_data_config(self) -> None:
        assert self.train_df is not None and self.val_df is not None and self.test_df is not None
        logger.info("Data:")
        logger.info("------")
        logger.info("data_dir=%s", self.base)
        logger.info("small_version=%s", self.small_version)
        logger.info("product_col=%s", self.product_col)
        logger.info(
            "train_rows=%d val_rows=%d test_rows=%d",
            len(self.train_df),
            len(self.val_df),
            len(self.test_df),
        )
        logger.info("Training:")
        logger.info("------")
        logger.info("model=%s device=%s", self.model_name, self.device or "auto")
        logger.info(
            "epochs=%d batch_size=%d lr=%g",
            self.epochs,
            self.batch_size,
            self.lr,
        )
        logger.info(
            "warmup_steps=%d max_length=%d",
            self.warmup_steps,
            self.max_length,
        )
        logger.info(
            "eval_steps=%d eval_max_queries=%s",
            self.evaluation_steps,
            self.eval_max_queries,
        )
        logger.info("save_path=%s", self.save_path)
        if self.early_stopping_patience > 0:
            logger.info(
                "early_stopping_patience=%d", self.early_stopping_patience
            )
        logger.info(
            "val_frac=%g (val used for mid-training eval; test held out until end)",
            self.val_frac,
        )

    def _validate_train_columns(self) -> None:
        assert self.train_df is not None
        if "esci_label" not in self.train_df.columns:
            raise ValueError("train_df must have 'esci_label' (from ESCIDataLoader.load_esci)")
        if self.product_col not in self.train_df.columns:
            raise ValueError(f"train_df must have '{self.product_col}'")

    def _setup_model(self) -> None:
        self.model = create_model(
            self.model_name,
            max_length=self.max_length,
            device=self.device,
        )

    def _build_train_dataloader(self) -> DataLoader:
        assert self.train_df is not None
        return build_dataloader(
            self.train_df,
            product_col=self.product_col,
            batch_size=self.batch_size,
        )

    def _build_val_evaluator(self) -> SequentialEvaluator | None:
        assert self.val_df is not None
        if len(self.val_df) == 0 or self.evaluation_steps <= 0:
            return None
        esc_eval = ESCIMetricsEvaluator(
            self.val_df,
            product_col=self.product_col,
            max_queries=self.eval_max_queries,
            batch_size=self.batch_size,
        )
        return SequentialEvaluator([esc_eval])

    def _fit_model(
        self,
        train_dataloader: DataLoader,
        evaluator: SequentialEvaluator | None,
    ) -> None:
        assert self.model is not None
        self.model.fit(
            train_dataloader,
            evaluator=evaluator,
            epochs=self.epochs,
            loss_fct=torch.nn.MSELoss(),
            activation_fct=torch.nn.Identity(),
            warmup_steps=self.warmup_steps,
            optimizer_params={"lr": self.lr},
            evaluation_steps=self.evaluation_steps,
            output_path=self.output_path,
            save_best_model=True,
        )

    def _save_model(self) -> None:
        if not self.output_path:
            return
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        assert self.model is not None
        self.model.save(self.output_path)
        logger.info("Reranker saved to %s", self.output_path)

    def _run_final_eval(self) -> None:
        assert self.test_df is not None and self.model is not None
        if len(self.test_df) == 0:
            return
        logger.info("------")
        logger.info("Final eval on held-out test set:")
        test_evaluator = ESCIMetricsEvaluator(
            self.test_df,
            product_col=self.product_col,
            max_queries=self.eval_max_queries,
            batch_size=self.batch_size,
        )
        test_evaluator(self.model, output_path=None, epoch=-1, steps=-1)
        m = test_evaluator.last_metrics
        logger.info(
            "  nDCG=%.4f MRR=%.4f MAP=%.4f Recall@10=%.4f",
            m["ndcg"],
            m["mrr"],
            m["map"],
            m["recall"],
        )


def main() -> int:
    """CLI entrypoint: load config from YAML and run training."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    p = argparse.ArgumentParser(
        description="Train cross-encoder reranker (ESCI approach)"
    )
    p.add_argument(
        "--config", default="configs/reranker.yaml", help="Path to YAML config"
    )
    args = p.parse_args()

    config_path = REPO_ROOT / args.config
    configs = load_config(config_path, DEFAULTS)

    if CrossEncoder is None or InputExample is None:
        raise ImportError("sentence-transformers is required for train_reranker")

    trainer = RerankerTrainer(
        data_dir=configs.get("data_dir"),
        model_name=configs.get("model_name"),
        product_col=configs.get("product_col"),
        save_path=configs.get("save_path"),
        epochs=configs.get("epochs"),
        batch_size=configs.get("batch_size"),
        lr=configs.get("lr"),
        warmup_steps=configs.get("warmup_steps"),
        max_length=configs.get("max_length"),
        evaluation_steps=configs.get("evaluation_steps"),
        eval_max_queries=configs.get("eval_max_queries"),
        small_version=configs.get("small_version", False),
        device=configs.get("device"),
        early_stopping_patience=configs.get("early_stopping_patience"),
        val_frac=configs.get("val_frac"),
    )
    trainer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
