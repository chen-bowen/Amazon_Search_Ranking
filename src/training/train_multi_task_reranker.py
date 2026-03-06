"""
Train multi-task learning ESCI model: Task 1 (query-product ranking),
Task 2 (4-class E/S/C/I), Task 3 (substitute identification).
Shared encoder with three heads; combined loss with configurable task weights.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from src.constants import (
    DATA_DIR,
    ESCI_LABEL2GAIN,
    ESCI_LABEL2ID,
    MODEL_CACHE_DIR,
    MULTI_TASK_RERANKER_DEFAULTS,
    REPO_ROOT,
)
from src.utils import clear_torch_cache, load_config
from src.data.load_data import ESCIDataLoader
from src.eval.evaluator import ESCIMetricsEvaluator, evaluate_classification_tasks
from src.models.multi_task_reranker import MultiTaskReranker

logger = logging.getLogger(__name__)


class MultiTaskDataset(Dataset):
    """
    Dataset of (query, product) pairs with multi-task learning targets:
    gain (Task 1: query-product ranking),
    class_id (Task 2: 4-class E/S/C/I),
    is_substitute (Task 3: substitute identification).
    """

    def __init__(
        self,
        pairs: list,
        gains: list,
        class_ids: list,
        is_substitute: list,
        tokenizer,
        max_length: int = 512,
    ) -> None:
        self.pairs = pairs
        self.gains = gains
        self.class_ids = class_ids
        self.is_substitute = is_substitute
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict:
        # Tokenize a single (query, product_text) pair into model inputs. We keep tokenization
        # inside the Dataset so the DataLoader yields ready-to-train tensors.
        enc = self.tokenizer(
            [self.pairs[i]],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        out = {k: v.squeeze(0) for k, v in enc.items()}
        # Multi-task targets:
        # - gain: graded relevance for ranking regression (Task 1)
        # - class_id: E/S/C/I id for 4-way classification (Task 2)
        # - is_substitute: 1 iff label is S (Task 3)
        out["gain"] = torch.tensor(self.gains[i], dtype=torch.float)
        out["class_id"] = torch.tensor(self.class_ids[i], dtype=torch.long)
        out["is_substitute"] = torch.tensor(float(self.is_substitute[i]), dtype=torch.float)
        return out


def build_multi_task_dataloader(
    train_df: pd.DataFrame,
    tokenizer,
    *,
    product_col: str,
    batch_size: int,
    max_length: int = 512,
) -> DataLoader:
    """
    Build DataLoader with (query, product_text, gain, class_id, is_substitute)
    per sample for multi-task learning (Task 1, Task 2, Task 3).
    Tokenization runs in MultiTaskDataset.__getitem__.
    """
    # Build inputs and labels from the ESCI dataframe. Targets are derived from esci_label:
    # - Task 1: regression gain for nDCG
    # - Task 2: 4-class id (E/S/C/I)
    # - Task 3: substitute binary (S vs non-S)
    pairs = []
    gains = []
    class_ids = []
    is_substitute = []
    for _, row in train_df.iterrows():
        label = str(row["esci_label"])
        pairs.append([str(row["query"]), str(row[product_col])])
        gains.append(ESCI_LABEL2GAIN.get(label, 0.0))
        class_ids.append(ESCI_LABEL2ID.get(label, 3))  # default I
        is_substitute.append(1.0 if label == "S" else 0.0)

    dataset = MultiTaskDataset(pairs, gains, class_ids, is_substitute, tokenizer, max_length=max_length)
    return DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        drop_last=True,
    )


class MultiTaskEvalWrapper:
    """
    Thin wrapper so ESCIMetricsEvaluator (expects model.predict -> scores only)
    works with multi-task learning reranker.
    """

    def __init__(self, model: MultiTaskReranker) -> None:
        self.model = model

    @property
    def device(self) -> torch.device:
        return self.model.device

    def predict(self, texts: list, batch_size: int = 32, show_progress_bar: bool = False) -> list:
        scores, _, _ = self.model.predict(texts, batch_size=batch_size, show_progress_bar=show_progress_bar)
        return scores


class MultiTaskTrainer:
    """Orchestrates training and evaluation of the multi-task reranker."""

    def __init__(
        self,
        *,
        data_dir: Path | str | None,
        model_name: str,
        product_col: str,
        save_path: str | Path | None,
        epochs: int,
        batch_size: int,
        max_length: int,
        lr: float,
        warmup_steps: int,
        task_weight_ranking: float,
        task_weight_esci: float,
        task_weight_substitute: float,
        evaluation_steps: int,
        eval_max_queries: int | None,
        small_version: bool,
        device: str | None,
        val_frac: float,
        recall_at: int,
    ) -> None:
        self.data_dir = data_dir
        self.model_name = model_name
        self.product_col = product_col
        self.save_path = save_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_length = max_length
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.task_weight_ranking = task_weight_ranking
        self.task_weight_esci = task_weight_esci
        self.task_weight_substitute = task_weight_substitute
        self.evaluation_steps = evaluation_steps
        self.eval_max_queries = eval_max_queries
        self.small_version = small_version
        self.device = device
        self.val_frac = val_frac
        self.recall_at = recall_at

        self.base = Path(self.data_dir or DATA_DIR)
        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None
        self.model: MultiTaskReranker | None = None
        self.train_dl: DataLoader | None = None
        self.evaluator: ESCIMetricsEvaluator | None = None
        self.output_path = str(self.save_path) if self.save_path else None
        self.best_ndcg: float | None = None
        self.global_step = 0
        self.opt: torch.optim.Optimizer | None = None
        self.sched: torch.optim.lr_scheduler.LRScheduler | None = None

    def run(self) -> MultiTaskReranker:
        self._load_splits()
        self._maybe_select_device()
        self._setup_model()
        self._setup_dataloader()
        self._setup_output_dir()
        self._setup_evaluator()
        self._setup_optim()
        self._train_epochs()
        self._save_final_checkpoint()
        self._run_test_eval()
        return self.model  # type: ignore[return-value]

    def _load_splits(self) -> None:
        loader = ESCIDataLoader(data_dir=self.base, small_version=self.small_version)
        train_df, val_df, test_df = loader.prepare_train_val_test(val_frac=self.val_frac)
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        logger.info(
            "Data: data_dir=%s train_rows=%d val_rows=%d test_rows=%d",
            self.base,
            len(train_df),
            len(val_df),
            len(test_df),
        )

    def _maybe_select_device(self) -> None:
        if self.device is None and torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("Using MPS (Apple Silicon GPU) for training.")

    def _setup_model(self) -> None:
        self.model = MultiTaskReranker(
            model_name=self.model_name,
            max_length=self.max_length,
            device=self.device,
            cache_folder=MODEL_CACHE_DIR,
        )

    def _setup_dataloader(self) -> None:
        assert self.train_df is not None and self.model is not None
        self.train_dl = build_multi_task_dataloader(
            self.train_df,
            self.model.tokenizer,
            product_col=self.product_col,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )

    def _setup_output_dir(self) -> None:
        if not self.output_path:
            return
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

    def _setup_evaluator(self) -> None:
        assert self.val_df is not None
        if len(self.val_df) == 0 or self.evaluation_steps <= 0:
            self.evaluator = None
            return
        self.evaluator = ESCIMetricsEvaluator(
            self.val_df,
            product_col=self.product_col,
            max_queries=self.eval_max_queries,
            batch_size=self.batch_size,
            recall_at_k=self.recall_at,
        )

    def _setup_optim(self) -> None:
        assert self.model is not None and self.train_dl is not None
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=0.01,
        )
        num_steps = len(self.train_dl) * self.epochs
        self.sched = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=num_steps,
        )

    def _train_epochs(self) -> None:
        assert self.train_dl is not None and self.model is not None
        for epoch in range(self.epochs):
            self._train_one_epoch(epoch)

    def _train_one_epoch(self, epoch: int) -> None:
        assert self.train_dl is not None and self.model is not None
        self.model.train()
        pbar = tqdm(self.train_dl, desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch")
        for batch in pbar:
            loss = self._train_step(batch)
            pbar.set_postfix({"loss": f"{loss:.4f}"})
            self.global_step += 1
            if self._should_evaluate():
                self._run_validation(epoch)

    def _train_step(self, batch: dict) -> float:
        assert self.model is not None and self.opt is not None and self.sched is not None
        dev = self.model.device
        gain = batch["gain"].to(dev)
        class_id = batch["class_id"].to(dev)
        is_sub = batch["is_substitute"].to(dev)
        token_type_ids = batch.get("token_type_ids")
        scores, esci_logits, sub_logits = self.model(
            batch["input_ids"].to(dev),
            batch["attention_mask"].to(dev),
            token_type_ids.to(dev) if token_type_ids is not None else None,
        )
        loss = self._compute_loss(gain, class_id, is_sub, scores, esci_logits, sub_logits)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self.sched.step()
        return float(loss.item())

    def _compute_loss(
        self,
        gain: torch.Tensor,
        class_id: torch.Tensor,
        is_sub: torch.Tensor,
        scores: torch.Tensor,
        esci_logits: torch.Tensor,
        sub_logits: torch.Tensor,
    ) -> torch.Tensor:
        l1 = F.mse_loss(scores, gain)
        l2 = F.cross_entropy(esci_logits, class_id)
        l3 = F.binary_cross_entropy_with_logits(sub_logits, is_sub)
        return (
            self.task_weight_ranking * l1
            + self.task_weight_esci * l2
            + self.task_weight_substitute * l3
        )

    def _should_evaluate(self) -> bool:
        return (
            self.evaluator is not None
            and self.evaluation_steps > 0
            and self.global_step % self.evaluation_steps == 0
        )

    def _run_validation(self, epoch: int) -> None:
        assert self.model is not None and self.evaluator is not None and self.val_df is not None
        self.model.eval()
        clear_torch_cache()
        ndcg = self.evaluator(
            MultiTaskEvalWrapper(self.model),
            output_path=None,
            epoch=epoch,
            steps=self.global_step,
        )
        evaluate_classification_tasks(
            self.model,
            self.val_df,
            product_col=self.product_col,
            max_queries=self.eval_max_queries,
            batch_size=self.batch_size,
            split_name="val",
        )
        self.model.train()
        if self.best_ndcg is None or ndcg > self.best_ndcg:
            self.best_ndcg = ndcg
            self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        if not self.output_path or self.model is None:
            return
        self.model.save(self.output_path)
        logger.info("Save model to %s", self.output_path)

    def _save_final_checkpoint(self) -> None:
        if not self.output_path or self.model is None:
            return
        self.model.save(self.output_path)
        logger.info("Multi-task learning reranker saved to %s", self.output_path)

    def _run_test_eval(self) -> None:
        assert self.test_df is not None and self.model is not None
        if len(self.test_df) == 0:
            return
        logger.info("------")
        logger.info("Final eval on held-out test set:")
        test_eval = ESCIMetricsEvaluator(
            self.test_df,
            product_col=self.product_col,
            max_queries=self.eval_max_queries,
            batch_size=self.batch_size,
            recall_at_k=self.recall_at,
        )
        test_eval(MultiTaskEvalWrapper(self.model), output_path=None, epoch=-1, steps=-1)
        m = test_eval.last_metrics
        logger.info(
            "  [Task 1] nDCG=%.4f MRR=%.4f MAP=%.4f Recall@%d=%.4f",
            m["ndcg"],
            m["mrr"],
            m["map"],
            self.recall_at,
            m["recall"],
        )
        evaluate_classification_tasks(
            self.model,
            self.test_df,
            product_col=self.product_col,
            max_queries=self.eval_max_queries,
            batch_size=self.batch_size,
            split_name="test",
        )


def main() -> int:
    """
    CLI entrypoint: load config from YAML and run multi-task learning training.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Train multi-task learning ESCI reranker (ranking + 4-class + substitute)")
    p.add_argument(
        "--config",
        default="configs/multi_task_reranker.yaml",
        help="Path to YAML config",
    )
    args = p.parse_args()
    config_path = REPO_ROOT / args.config
    configs = load_config(config_path, MULTI_TASK_RERANKER_DEFAULTS)

    trainer = MultiTaskTrainer(
        data_dir=configs.get("data_dir"),
        model_name=configs.get("model_name"),
        product_col=configs.get("product_col"),
        save_path=configs.get("save_path"),
        epochs=configs.get("epochs"),
        batch_size=configs.get("batch_size"),
        max_length=configs.get("max_length"),
        lr=configs.get("lr"),
        warmup_steps=configs.get("warmup_steps"),
        task_weight_ranking=configs.get("task_weight_ranking"),
        task_weight_esci=configs.get("task_weight_esci"),
        task_weight_substitute=configs.get("task_weight_substitute"),
        evaluation_steps=configs.get("evaluation_steps"),
        eval_max_queries=configs.get("eval_max_queries"),
        small_version=configs.get("small_version", False),
        device=configs.get("device"),
        val_frac=configs.get("val_frac"),
        recall_at=configs.get("recall_at"),
    )
    trainer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
