"""
Training loop: load relevant (query, product) pairs, in-batch negatives, self-adversarial loss.
"""

from __future__ import annotations

import argparse
import logging
import random
import warnings
from collections import deque
from pathlib import Path

import pandas as pd
import torch
import yaml
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import Dataset, DataLoader, Sampler

from src.constants import DATA_DIR, DEFAULT_MODEL_NAME, REPO_ROOT
from src.data.load_data import prepare_train_test
from src.data.query_augment import augment_query
from src.models.retriever import BiEncoderRetriever
from src.utils import setup_colored_logging
from src.training.loss import contrastive_loss_with_reweighting
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class QueryProductDataset(Dataset):
    """Dataset of (query, product_text) pairs; positive pairs (relevance >= 2, i.e. E, S, C)."""

    def __init__(
        self,
        df: pd.DataFrame,
        min_relevance: int = 2,
        use_query_augment: bool = True,
        augment_prob: float = 0.25,
    ):
        # Filter to positive pairs: relevance >= 2 (E=4, S=2, C=3; excludes I=1)
        pos = df[df["relevance"] >= min_relevance][["query_id", "query", "product_text"]].drop_duplicates()
        self.df = pos.reset_index(drop=True)
        self.use_query_augment = use_query_augment
        self.augment_prob = augment_prob
        # Group indices by query_id for unique-query batching (avoids in-batch false negatives)
        self._qid_to_indices: dict[str, list[int]] = {}
        for i, qid in enumerate(self.df["query_id"].astype(str)):
            # Append each row index to its query's list (one query can have multiple products)
            self._qid_to_indices.setdefault(qid, []).append(i)
        self._query_ids = list(self._qid_to_indices.keys())

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> tuple[str, str]:
        row = self.df.iloc[i]
        query = str(row["query"])
        # Optionally substitute partial query with synonym (data augmentation)
        if self.use_query_augment:
            query = augment_query(query, prob=self.augment_prob)
        return query, str(row["product_text"])


class UniqueQueryBatchSampler(Sampler[list[int]]):
    """
    Ensures at most one (query, product) pair per query per batch.
    Fixes in-batch false negatives: when query A has products p1,p2, both relevant,
    we must not treat (A,p2) as negative when (A,p1) is the positive.

    Iterates over ALL (query, product) pairs so the same query can appear in
    different batches with different products. Each batch has unique queries only.
    """

    def __init__(
        self,
        qid_to_indices: dict[str, list[int]],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.qid_to_indices = qid_to_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        # Flatten to (dataset_index, query_id) for every (query, product) pair
        self._pairs: list[tuple[int, str]] = [(idx, qid) for qid, indices in qid_to_indices.items() for idx in indices]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _fill_from_deferred(
        self,
        batch: list[int],
        seen_qids: set[str],
        deferred: list[tuple[int, str]],
    ) -> None:
        """
        Move deferred items into batch until full or saturated.

        Deferred = pairs we couldn't add earlier because their query was already
        in the batch. When we start a new batch, we try to drain them. Saturated
        when all deferred share a qid already in the current batch.

        Parameters
        ----------
        batch : list[int]
            List of indices to fill.
        seen_qids : set[str]
            Set of query IDs already in batch.
        deferred : list[tuple[int, str]]
            Queue of (index, query_id) pairs waiting for a batch with room.
        """
        # Drain deferred items until batch is full or saturated
        while deferred and len(batch) < self.batch_size:
            d_idx, d_qid = deferred.pop(0)
            if d_qid not in seen_qids:
                batch.append(d_idx)
                seen_qids.add(d_qid)
            else:
                deferred.append((d_idx, d_qid))
                if all(q in seen_qids for _, q in deferred):
                    break

    def __iter__(self):
        """
        Iterate over batches of (query, product) pairs. Could be used as sliceable iterator.

        Returns
        -------
        Iterator[list[int]]
            Iterator of lists of indices.
        """
        rng = random.Random(self.seed + self.epoch)
        pairs = list(self._pairs)
        if self.shuffle:
            rng.shuffle(pairs)

        batch: list[int] = []
        seen_qids: set[str] = set()
        # Pairs we couldn't add because their query is already in the batch; drained when we start a new batch
        deferred: list[tuple[int, str]] = []

        # Iterate over pairs and add to batch, ensuring one (query, product) per query per batch
        for idx, qid in pairs:
            if qid in seen_qids:
                deferred.append((idx, qid))
                continue
            batch.append(idx)
            seen_qids.add(qid)
            # If batch is full, yield it and reset
            if len(batch) >= self.batch_size:
                yield batch
                batch, seen_qids = [], set()
                self._fill_from_deferred(batch, seen_qids, deferred)

        # If there are any deferred items, yield the batch and reset
        while deferred:
            if batch:
                yield batch
            batch, seen_qids = [], set()
            self._fill_from_deferred(batch, seen_qids, deferred)

        if batch:
            yield batch

    def __len__(self) -> int:
        """
        Number of batches in the dataset. Could be used as len(batch_sampler).

        Returns
        -------
        int
            Number of batches.
        """
        n_pairs = len(self._pairs)
        # Ceiling division; at least 1 batch
        return max(1, (n_pairs + self.batch_size - 1) // self.batch_size)


def resolve_device(preferred: str | torch.device | None = None) -> torch.device:
    """
    Resolve a training device with preference order:
    1) explicit `preferred` ("mps", "cuda", "cpu" or torch.device),
    2) available CUDA,
    3) available MPS,
    4) CPU.
    """
    # Normalize string vs torch.device for uniform handling
    if isinstance(preferred, torch.device):
        preferred_str = preferred.type
    else:
        preferred_str = str(preferred) if preferred is not None else None

    # Honor explicit request when possible
    if preferred_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred_str == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred_str == "cpu":
        return torch.device("cpu")

    # Automatic fallback order
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_ir_eval_data(eval_df: pd.DataFrame, min_relevance: int = 2) -> tuple[dict, dict, dict]:
    """
    Build (queries, corpus, relevant_docs) for sentence_transformers InformationRetrievalEvaluator.
    Corpus = unique product_id -> product_text; relevant_docs = query_id -> set of product_ids with relevance >= min_relevance.

    Parameters
    ----------
    - eval_df : pd.DataFrame
        DataFrame with query, product_id, product_text, and relevance columns.
    - min_relevance : int
        Minimum relevance score to consider as relevant.

    Returns
    -------
    - queries : dict[str, str]
        Dictionary of query_id -> query.
    - corpus : dict[str, str]
        Dictionary of product_id -> product_text.
    - relevant_docs : dict[str, set[str]]
        Dictionary of query_id -> set of product_ids with relevance >= min_relevance.
    """
    if "product_id" not in eval_df.columns:
        raise ValueError("eval_df must have 'product_id' for InformationRetrievalEvaluator (e.g. from load_esci).")
    # One query string per query_id (first occurrence)
    queries = eval_df.groupby("query_id")["query"].first().astype(str).to_dict()
    queries = {str(k): v for k, v in queries.items()}
    corpus_df = eval_df[["product_id", "product_text"]].drop_duplicates("product_id")
    corpus = {str(pid): str(text) for pid, text in zip(corpus_df["product_id"], corpus_df["product_text"])}
    # Map query_id -> set of product_ids with relevance >= min_relevance
    relevant_docs: dict[str, set[str]] = {}
    for _, row in eval_df.iterrows():
        if row["relevance"] >= min_relevance:
            qid = str(row["query_id"])
            pid = str(row["product_id"])
            relevant_docs.setdefault(qid, set()).add(pid)
    return queries, corpus, relevant_docs


def subsample_ir_eval(
    queries: dict[str, str],
    corpus: dict[str, str],
    relevant_docs: dict[str, set[str]],
    max_queries: int = 2000,
    max_corpus: int = 50000,
) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    """
    Subsample IR eval so mid-epoch eval is faster but metrics stay meaningful.

    - Subsample queries to at most max_queries.
    - Build corpus as: all relevant product_ids for those queries, then fill to
      max_corpus with random other products. This ensures every sampled query has
      its relevant docs in the corpus (avoids artificially zero metrics).
    """
    rng = random.Random(42)

    # Subsample queries (keeps eval fast while preserving metric validity)
    query_ids = list(queries.keys())
    if len(query_ids) > max_queries:
        keep_q = set(rng.sample(query_ids, max_queries))
        queries = {qid: text for qid, text in queries.items() if qid in keep_q}
        relevant_docs = {qid: docs for qid, docs in relevant_docs.items() if qid in keep_q}

    # Corpus must include all relevant docs for sampled queries, then fill to max_corpus
    must_have = set()
    for docs in relevant_docs.values():
        must_have |= docs
    rest_ids = [cid for cid in corpus if cid not in must_have]
    rng.shuffle(rest_ids)
    # Fill remaining slots with random products so corpus size hits max_corpus
    need = max(0, max_corpus - len(must_have))
    keep_c = must_have | set(rest_ids[:need])
    corpus = {cid: corpus[cid] for cid in keep_c}
    # Restrict relevant_docs to products in corpus; drop queries with no relevant docs left
    relevant_docs = {qid: (docs & keep_c) for qid, docs in relevant_docs.items() if (docs & keep_c)}
    queries = {qid: text for qid, text in queries.items() if qid in relevant_docs}

    return queries, corpus, relevant_docs


def build_ir_evaluators(
    eval_df: pd.DataFrame | None,
    batch_size: int,
    *,
    max_mid_queries: int = 2000,
    max_mid_corpus: int = 50_000,
) -> tuple[InformationRetrievalEvaluator | None, InformationRetrievalEvaluator | None]:
    """
    Construct two IR evaluators:

    - A fast, subsampled evaluator for mid-epoch feedback.
    - A full evaluator over the entire eval_df for end-of-epoch metrics.
    """
    if eval_df is None or len(eval_df) == 0:
        return None, None

    # Full IR artifacts.
    queries_full, corpus_full, relevant_full = build_ir_eval_data(eval_df)
    orig_q, orig_c = len(queries_full), len(corpus_full)

    # Subsampled artifacts for mid-epoch eval.
    queries_mid, corpus_mid, relevant_mid = subsample_ir_eval(
        dict(queries_full),
        dict(corpus_full),
        dict(relevant_full),
        max_queries=max_mid_queries,
        max_corpus=max_mid_corpus,
    )

    # Mid-epoch evaluator on a subsample (faster).
    ir_eval_mid = InformationRetrievalEvaluator(
        queries=queries_mid,
        corpus=corpus_mid,
        relevant_docs=relevant_mid,
        name="esci-eval-subsample",
        show_progress_bar=True,
        write_csv=False,
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 10],
        precision_recall_at_k=[10],
        map_at_k=[10, 100],
        batch_size=min(64, batch_size),
    )

    # Full evaluator for end-of-epoch metrics.
    ir_eval_full = InformationRetrievalEvaluator(
        queries=queries_full,
        corpus=corpus_full,
        relevant_docs=relevant_full,
        name="esci-eval",
        show_progress_bar=True,
        write_csv=False,
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 10],
        precision_recall_at_k=[10],
        map_at_k=[10, 100],
        batch_size=min(64, batch_size),
    )

    logger.info(
        "  eval              : %d→%d queries, %d→%d corpus docs (mid-epoch subsample); full eval on all.",
        orig_q,
        len(queries_mid),
        orig_c,
        len(corpus_mid),
    )
    return ir_eval_mid, ir_eval_full


def collate_query_product(batch: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    # Unzip [(q1,p1), (q2,p2), ...] -> ([q1,q2,...], [p1,p2,...])
    """
    Collate function: convert list of (query, product) tuples into separate lists.

    Parameters
    ----------
    batch : list[tuple[str, str]]
        List of (query, product) tuples.

    Returns
    -------
    tuple[list[str], list[str]]
        Tuple of lists of queries and products.

    Notes
    -----
    Unzip: [(q1,p1), (q2,p2), ...] -> ([q1,q2,...], [p1,p2,...])
    """
    # zip(*batch) transposes list of tuples into separate lists
    queries, products = zip(*batch)
    return list(queries), list(products)


def run_training(
    train_df: pd.DataFrame | None = None,
    data_dir: Path | str | None = None,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    shared_tower: bool = False,
    batch_size: int = 64,
    epochs: int = 3,
    lr: float = 2e-5,
    reweight_hard: bool = True,
    hard_weight_power: float = 1.0,
    temperature: float = 1.0,
    use_query_augment: bool = True,
    augment_prob: float = 0.25,
    device: str | torch.device = "cuda",
    save_path: Path | str | None = None,
    use_saved_splits: bool = False,
    small_version: bool = False,
) -> BiEncoderRetriever:
    """
    Main training function: load data, create model, train with contrastive loss.

    Notation (also used in comments):
    - B: batch size = number of (query, product) pairs per step.
    - D: embedding dimension produced by the encoder.

    Parameters
    ----------
    - train_df : pd.DataFrame
        DataFrame with query, product_text, and relevance columns.
    - data_dir : Path | str | None
        Directory with esci_train.parquet and esci_test.parquet.
    - model_name : str
        Name of the model to use.
    - shared_tower : bool
        Whether to use a shared tower for both query and product.
    - batch_size : int
        Number of (query, product) pairs per step.
    - epochs : int
        Number of epochs to train for.
    - lr : float
        Learning rate.
    - reweight_hard : bool
        Whether to use hard-negative reweighting.
    - hard_weight_power : float
        Power of the hard-negative reweighting.
    - temperature : float
        Temperature of the softmax function.
    - use_query_augment : bool
        Whether to use query augmentation.
    - augment_prob : float
        Probability of augmenting a query.
    - device : str | torch.device
        Device to train on.
    - save_path : Path | str | None
        Path to save the model.
    - use_saved_splits : bool
        Whether to use the saved train and test splits.

    Returns
    -------
    - model : TwoTowerEncoder
        Trained TwoTowerEncoder model.
    """
    eval_df: pd.DataFrame | None = None
    base = Path(data_dir or DATA_DIR)
    # Load from parquet if available, else prepare from raw ESCI
    if train_df is None:
        train_path = base / "esci_train.parquet"
        use_saved = use_saved_splits and train_path.exists()
        if use_saved:
            train_df = pd.read_parquet(train_path)
            eval_path = base / "esci_test.parquet"
            if eval_path.exists():
                eval_df = pd.read_parquet(eval_path)
        else:
            # Load raw ESCI from base data directory (expects parquets in `data/`).
            train_df, eval_df = prepare_train_test(data_dir=base, small_version=small_version)

    device = resolve_device(device)

    # Bi-encoder: separate query/product encoders; outputs embeddings [B, D]
    model = BiEncoderRetriever(model_name=model_name, shared=shared_tower, normalize=True)
    model = model.to(device)
    # Wrap the positive (query, product_text) pairs in a Dataset so we can batch them.
    dataset = QueryProductDataset(
        train_df,
        use_query_augment=use_query_augment,
        augment_prob=augment_prob,
    )

    # Batch sampler ensures one (query, product) per query per batch (avoids in-batch false negatives).
    batch_sampler = UniqueQueryBatchSampler(
        dataset._qid_to_indices,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_query_product,
        num_workers=0,
    )

    # Pretty-print training configuration and dataset statistics.
    logger.info("Training configuration:")
    logger.info("  model_name       : %s", model_name)
    logger.info("  shared_tower     : %s", shared_tower)
    logger.info("  batch_size       : %d", batch_size)
    logger.info("  epochs           : %d", epochs)
    logger.info("  lr               : %.2e", lr)
    logger.info("  reweight_hard    : %s", reweight_hard)
    logger.info("  hard_weight_power: %.2f", hard_weight_power)
    logger.info("  temperature      : %.3f (softmax sharpness; <1 sharper, >1 softer)", temperature)
    logger.info("  use_query_augment: %s", use_query_augment)
    logger.info("  augment_prob     : %.2f", augment_prob)
    logger.info("  device           : %s", device)
    logger.info("  use_saved_splits : %s (False = load full ESCI from raw; True = use esci_*.parquet if present)", use_saved_splits)
    logger.info("  small_version    : %s (True = Task 1 reduced set, ~48k queries; False = full ~130k)", small_version)
    logger.info("Dataset statistics:")
    logger.info("  rows (raw)       : %d", len(train_df))
    logger.info("  rows (positives) : %d", len(dataset))
    logger.info("  unique queries   : %d", len(dataset._qid_to_indices))
    logger.info("  pairs per epoch  : %d (all pairs used; one query per batch to avoid in-batch false negatives)", len(dataset))

    # Optional full-ranking evaluators on the held-out eval_df (nDCG@10, MRR@10, etc.).
    # We use a subsampled evaluator mid-epoch for cheap feedback, and a full evaluator
    # at the end of each epoch for accurate metrics.
    try:
        ir_evaluator_mid, ir_evaluator_full = build_ir_evaluators(eval_df, batch_size=batch_size)
    except Exception as e:
        logger.warning("  eval              : skipped (build IR evaluators failed: %s)", e)
        ir_evaluator_mid, ir_evaluator_full = None, None

    # Optimizer over all encoder parameters.
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # Warmup (10% of steps) then cosine decay to 0
    total_steps = max(len(dataloader) * epochs, 1)
    warmup_steps = max(int(0.1 * total_steps), 1)
    cosine_steps = max(total_steps - warmup_steps, 1)
    # Linear warmup from 0.001x to 1x, then cosine decay
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, total_iters=warmup_steps)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cosine_steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
    model.train()
    loss_window = 50
    global_step = 0

    # Global progress bar over all epochs / steps.
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * epochs
    progress = tqdm(
        total=total_steps,
        desc="Training",
        leave=False,
        position=0,
    )

    for epoch in range(epochs):
        # Different shuffle order per epoch (used by UniqueQueryBatchSampler)
        batch_sampler.set_epoch(epoch)
        total_loss = 0.0  # running sum of loss over this epoch
        n_batches = 0  # number of batches seen in this epoch
        grad_norm_sum = 0.0  # running sum of gradient norms (for avg_grad_norm)
        loss_smooth = 0.0  # online mean of loss over the epoch

        # Rolling window for "last N" loss average
        loss_last = deque(maxlen=loss_window)

        # Stats line (no progress bar) below main bar
        stats_bar = tqdm(
            total=1,
            bar_format="{desc}",
            leave=False,
            position=1,
        )
        for queries, products in dataloader:
            opt.zero_grad()
            # Forward: encode queries and products -> [B, D] embeddings
            q_emb, p_emb = model(
                query_strings=queries,
                product_strings=products,
                device=device,
            )

            # Contrastive loss (in-batch negatives) with optional hard-negative reweighting
            loss = contrastive_loss_with_reweighting(
                q_emb,
                p_emb,
                reweight_hard=reweight_hard,
                hard_weight_power=hard_weight_power,
                temperature=temperature,
            )
            loss.backward()

            # L2 norm of all gradients (for logging; no gradient clipping)
            total_grad_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_sq += p.grad.detach().pow(2).sum().item()
            grad_norm = total_grad_sq**0.5

            opt.step()
            scheduler.step()
            global_step += 1

            # Update running loss and gradient norm sums.
            loss_value = loss.item()
            total_loss += loss_value
            n_batches += 1
            grad_norm_sum += grad_norm
            loss_last.append(loss_value)

            # Online exponential moving average of loss
            loss_smooth = loss_smooth + (loss_value - loss_smooth) / min(n_batches, loss_window)
            avg_last_50 = sum(loss_last) / len(loss_last) if loss_last else 0.0
            current_lr_step = opt.param_groups[0]["lr"]

            stats_bar.set_description_str(
                f"Epoch {epoch + 1}/{epochs} | loss={loss_smooth:.4f},  last50={avg_last_50:.4f}, |g|={grad_norm:.4f}, lr={current_lr_step:.2e}"
            )
            progress.update(1)
            # Mid-epoch IR eval on subsample (every 5000 batches)
            if ir_evaluator_mid is not None and n_batches % 5000 == 0:
                progress.write(f"Information Retrieval eval (mid-epoch, subsample): epoch {epoch + 1}, step {global_step} ...")
                eval_results_mid = ir_evaluator_mid(model, epoch=epoch + 1, steps=global_step)
                # Keys from sentence_transformers IR evaluator (name + metric)
                ndcg_key = "esci-eval_cosine_ndcg@10"
                mrr_key = "esci-eval_cosine_mrr@10"
                acc10_key = "esci-eval_cosine_accuracy@10"
                recall10_key = "esci-eval_cosine_recall@10"
                map10_key = "esci-eval_cosine_map@10"
                logger.info(
                    "Eval(step %d) nDCG@10=%.4f  MRR@10=%.4f  Acc@10=%.4f  Recall@10=%.4f  MAP@10=%.4f",
                    global_step,
                    eval_results_mid.get(ndcg_key, 0.0),
                    eval_results_mid.get(mrr_key, 0.0),
                    eval_results_mid.get(acc10_key, 0.0),
                    eval_results_mid.get(recall10_key, 0.0),
                    eval_results_mid.get(map10_key, 0.0),
                )
        stats_bar.close()
        avg_loss = total_loss / max(n_batches, 1)
        avg_grad_norm = grad_norm_sum / max(n_batches, 1)
        loss_last_50 = sum(loss_last) / len(loss_last) if loss_last else 0.0
        current_lr = opt.param_groups[0]["lr"]
        logger.info(
            "Epoch %d/%d avg_loss=%.4f loss_last_50=%.4f avg_grad_norm=%.4f lr=%.2e",
            epoch + 1,
            epochs,
            avg_loss,
            loss_last_50,
            avg_grad_norm,
            current_lr,
        )
        # End-of-epoch full IR eval (nDCG@10, MRR@10, etc.)
        if ir_evaluator_full is not None:
            tqdm.write(f"Information Retrieval eval (end of epoch {epoch + 1}/{epochs}, full eval set) ...")
            eval_results = ir_evaluator_full(model, epoch=epoch + 1)
            # Same key names as mid-epoch evaluator
            ndcg_key = "esci-eval_cosine_ndcg@10"
            mrr_key = "esci-eval_cosine_mrr@10"
            acc10_key = "esci-eval_cosine_accuracy@10"
            recall10_key = "esci-eval_cosine_recall@10"
            map10_key = "esci-eval_cosine_map@10"
            logger.info(
                "Eval       nDCG@10=%.4f  MRR@10=%.4f  Acc@10=%.4f  Recall@10=%.4f  MAP@10=%.4f",
                eval_results.get(ndcg_key, 0.0),
                eval_results.get(mrr_key, 0.0),
                eval_results.get(acc10_key, 0.0),
                eval_results.get(recall10_key, 0.0),
                eval_results.get(map10_key, 0.0),
            )
    progress.close()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "model_name": model_name}, save_path)
    return model


def main() -> int:
    """
    Command-line entry point: parse args, load config, run training.

    Defaults
    --------
    If a flag is omitted, values are resolved in this order:
    1. Command-line flag (if provided).
    2. `configs/train.yaml` (if the key exists).
    3. Hard-coded fallback:
       - data_dir: `DATA_DIR` (project `data/` directory).
       - model_name: `"all-MiniLM-L12-v2"`.
       - shared_tower: False.
       - batch_size: 64.
       - epochs: 3.
       - lr: 2e-5.
       - reweight_hard: True.
       - hard_weight_power: 1.0.
       - temperature: 0.01.
       - use_query_augment: True.
       - augment_prob: 0.25.
       - save_path: `"data/model.pt"`.
       - use_saved_splits: False (load full ESCI from raw; set True or pass --use-saved-splits to use parquets).
    """

    setup_colored_logging(
        level=logging.INFO,
        fmt="%(message)s",
        quiet_loggers=[
            "httpx",
            "httpcore",
            "urllib3",
            "huggingface_hub",
            "sentence_transformers",
        ],
    )
    # Suppress HuggingFace unauthenticated-request warnings
    warnings.filterwarnings(
        "ignore",
        message=".*unauthenticated requests to the HF Hub.*",
    )
    p = argparse.ArgumentParser(description="Train two-tower encoder on ESCI")
    p.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Optional YAML config file (default: configs/train.yaml)",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory with esci_train.parquet (default: data/ or value from config)",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="SentenceTransformer backbone (default: all-MiniLM-L12-v2)",
    )
    p.add_argument(
        "--shared",
        action="store_true",
        help="Use a shared encoder for both towers (default: separate towers)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: 64)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (default: 3)",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 2e-5)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Softmax temperature (logits / T). T < 1.0 sharper, T > 1.0 softer (default: 0.01)",
    )
    p.add_argument(
        "--loss-scale",
        type=float,
        default=None,
        help="Scale loss before backward for larger grad norms, e.g. 25 (default: 1.0)",
    )
    p.add_argument(
        "--no-reweight-hard",
        action="store_true",
        help="Disable self-adversarial hard-negative reweighting",
    )
    p.add_argument(
        "--hard-weight-power",
        type=float,
        default=None,
        help="Hard-negative weight power β (default: 1.0)",
    )
    p.add_argument(
        "--no-query-augment",
        action="store_true",
        help="Disable partial-query augmentation",
    )
    p.add_argument(
        "--augment-prob",
        type=float,
        default=None,
        help="Probability of substituting a partial query (default: 0.25)",
    )
    p.add_argument(
        "--save",
        type=str,
        default=None,
        help='Path to save model checkpoint (default: "data/model.pt")',
    )
    p.add_argument(
        "--use-saved-splits",
        action="store_true",
        help="Use esci_train.parquet / esci_test.parquet if present; default is to load full ESCI from raw",
    )
    p.add_argument(
        "--small-version",
        action="store_true",
        help="Use Task 1 reduced set (~48k queries); default is full ESCI (~130k)",
    )
    args = p.parse_args()
    cfg: dict = {}
    config_path = REPO_ROOT / args.config
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    # CLI overrides config overrides defaults
    data_dir = Path(args.data_dir or cfg.get("data_dir") or DATA_DIR)
    run_training(
        data_dir=data_dir,
        model_name=args.model_name or cfg.get("model_name", "all-MiniLM-L12-v2"),
        shared_tower=args.shared or cfg.get("shared_tower", False),
        batch_size=args.batch_size if args.batch_size is not None else cfg.get("batch_size", 64),
        epochs=args.epochs if args.epochs is not None else cfg.get("epochs", 3),
        lr=args.lr if args.lr is not None else cfg.get("lr", 2e-5),
        temperature=args.temperature if args.temperature is not None else cfg.get("temperature", 1.0),
        reweight_hard=not args.no_reweight_hard and cfg.get("reweight_hard", True),
        hard_weight_power=args.hard_weight_power if args.hard_weight_power is not None else cfg.get("hard_weight_power", 1.0),
        use_query_augment=not args.no_query_augment and cfg.get("use_query_augment", True),
        augment_prob=args.augment_prob if args.augment_prob is not None else cfg.get("augment_prob", 0.25),
        save_path=args.save or cfg.get("save_path", "data/model.pt"),
        use_saved_splits=args.use_saved_splits or cfg.get("use_saved_splits", False),
        small_version=args.small_version or cfg.get("small_version", False),
    )
    return 0


if __name__ == "__main__":
    main()
