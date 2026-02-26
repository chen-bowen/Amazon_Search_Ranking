"""
Training loop: load relevant (query, product) pairs, in-batch negatives, self-adversarial loss.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from collections import deque
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import Dataset, DataLoader

from src.constants import DATA_DIR, DEFAULT_MODEL_NAME, REPO_ROOT
from src.data.load_data import load_esci, prepare_train_test
from src.data.query_augment import augment_query
from src.models.two_tower import TwoTowerEncoder
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
        self.df = df[df["relevance"] >= min_relevance][["query", "product_text"]].drop_duplicates().reset_index(drop=True)
        self.use_query_augment = use_query_augment
        self.augment_prob = augment_prob

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> tuple[str, str]:
        row = self.df.iloc[i]
        query = str(row["query"])
        if self.use_query_augment:
            query = augment_query(query, prob=self.augment_prob)
        return query, str(row["product_text"])


def resolve_device(preferred: str | torch.device | None = None) -> torch.device:
    """
    Resolve a training device with preference order:
    1) explicit `preferred` ("mps", "cuda", "cpu" or torch.device),
    2) available CUDA,
    3) available MPS,
    4) CPU.
    """
    # Normalize string vs torch.device
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
    """
    eval_df = eval_df.copy()
    if "product_id" not in eval_df.columns:
        raise ValueError("eval_df must have 'product_id' for InformationRetrievalEvaluator (e.g. from load_esci).")
    queries = eval_df.groupby("query_id")["query"].first().astype(str).to_dict()
    queries = {str(k): v for k, v in queries.items()}
    corpus_df = eval_df[["product_id", "product_text"]].drop_duplicates("product_id")
    corpus = {str(pid): str(text) for pid, text in zip(corpus_df["product_id"], corpus_df["product_text"])}
    relevant_docs: dict[str, set[str]] = {}
    for _, row in eval_df.iterrows():
        if row["relevance"] >= min_relevance:
            qid = str(row["query_id"])
            pid = str(row["product_id"])
            relevant_docs.setdefault(qid, set()).add(pid)
    return queries, corpus, relevant_docs


def collate_query_product(batch: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
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
    queries, products = zip(*batch)  # Unzip: [(q1,p1), (q2,p2), ...] -> ([q1,q2,...], [p1,p2,...])
    return list(queries), list(products)  # Convert tuples to lists


def run_training(
    train_df: pd.DataFrame | None = None,
    data_dir: Path | str | None = None,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    shared_tower: bool = False,
    batch_size: int = 64,
    epochs: int = 3,
    lr: float = 2e-5,
    loss_scale: float = 1.0,
    reweight_hard: bool = True,
    hard_weight_power: float = 1.0,
    use_query_augment: bool = True,
    augment_prob: float = 0.25,
    device: str | torch.device = "cuda",
    save_path: Path | str | None = None,
    use_saved_splits: bool = False,
) -> TwoTowerEncoder:
    """
    Main training function: load data, create model, train with contrastive loss.

    Notation (also used in comments):
    - B: batch size = number of (query, product) pairs per step.
    - D: embedding dimension produced by the encoder.
    """
    eval_df: pd.DataFrame | None = None
    base = Path(data_dir or DATA_DIR)
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
            _df = load_esci(data_dir=base)
            train_df, eval_df = prepare_train_test(df=_df)

    # Resolve device (preferring explicit choice, then CUDA, then MPS, then CPU).
    device = resolve_device(device)

    # Two-tower encoder outputs query/product embeddings of shape [B, D].
    model = TwoTowerEncoder(model_name=model_name, shared=shared_tower, normalize=True)
    model = model.to(device)
    dataset = QueryProductDataset(
        train_df,
        use_query_augment=use_query_augment,
        augment_prob=augment_prob,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_query_product,
        num_workers=0,
        drop_last=True,
    )

    # Pretty-print training configuration and dataset statistics.
    logger.info("Training configuration:")
    logger.info("  model_name       : %s", model_name)
    logger.info("  shared_tower     : %s", shared_tower)
    logger.info("  batch_size       : %d", batch_size)
    logger.info("  epochs           : %d", epochs)
    logger.info("  lr               : %.2e", lr)
    logger.info("  loss_scale       : %.1f (grad_norm scales with this; use smaller lr if you increase it)", loss_scale)
    logger.info("  reweight_hard    : %s", reweight_hard)
    logger.info("  hard_weight_power: %.2f", hard_weight_power)
    logger.info("  use_query_augment: %s", use_query_augment)
    logger.info("  augment_prob     : %.2f", augment_prob)
    logger.info("  device           : %s", device)
    logger.info("  use_saved_splits : %s (False = load full ESCI from raw; True = use esci_*.parquet if present)", use_saved_splits)
    logger.info("Dataset statistics:")
    logger.info("  rows (raw)       : %d", len(train_df))
    logger.info("  rows (positives) : %d", len(dataset))

    ir_evaluator: InformationRetrievalEvaluator | None = None
    if eval_df is not None and len(eval_df) > 0:
        try:
            queries_ir, corpus_ir, relevant_docs_ir = build_ir_eval_data(eval_df)
            ir_evaluator = InformationRetrievalEvaluator(
                queries=queries_ir,
                corpus=corpus_ir,
                relevant_docs=relevant_docs_ir,
                name="esci-eval",
                show_progress_bar=False,
                write_csv=False,
                mrr_at_k=[10],
                ndcg_at_k=[10],
                accuracy_at_k=[1, 10],
                batch_size=min(64, batch_size),
            )
            logger.info(
                "  eval              : %d queries, %d corpus docs (InformationRetrievalEvaluator each epoch)",
                len(queries_ir),
                len(corpus_ir),
            )
        except Exception as e:
            logger.warning("  eval              : skipped (build IR evaluator failed: %s)", e)
            ir_evaluator = None

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    # Warmup + cosine LR schedule using standard PyTorch schedulers.
    total_steps = max(len(dataloader) * epochs, 1)
    warmup_steps = max(int(0.1 * total_steps), 1)
    cosine_steps = max(total_steps - warmup_steps, 1)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1e-3, total_iters=warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cosine_steps
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
    model.train()
    loss_window = 50
    global_step = 0
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        grad_norm_sum = 0.0
        loss_smooth = 0.0
        # Last N step losses for "loss past 50 steps" average.
        loss_last = deque(maxlen=loss_window)
        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=False,
            position=0,
        )
        # Second line: full metrics (no bar, description-only) so nothing is truncated.
        stats_bar = tqdm(
            total=1,
            bar_format="{desc}",
            leave=False,
            position=1,
        )
        for queries, products in progress:
            opt.zero_grad()
            q_emb, p_emb = model(
                query_strings=queries,
                product_strings=products,
                device=device,
            )
            loss = contrastive_loss_with_reweighting(
                q_emb,
                p_emb,
                reweight_hard=reweight_hard,
                hard_weight_power=hard_weight_power,
            )
            (loss * loss_scale).backward()

            # Compute gradient L2 norm for monitoring (no clipping).
            total_grad_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_sq += p.grad.detach().pow(2).sum().item()
            grad_norm = total_grad_sq**0.5

            opt.step()
            scheduler.step()
            global_step += 1
            loss_value = loss.item()
            total_loss += loss_value
            n_batches += 1
            grad_norm_sum += grad_norm
            loss_last.append(loss_value)

            loss_smooth = loss_smooth + (loss_value - loss_smooth) / min(n_batches, loss_window)
            avg_last_50 = sum(loss_last) / len(loss_last) if loss_last else 0.0
            current_lr_step = opt.param_groups[0]["lr"]
            stats_bar.set_description_str(
                f"Training Statistics:   [avg_loss={loss_smooth:.4f}  loss_last_50={avg_last_50:.4f}  grad_norm={grad_norm:.4f}  lr={current_lr_step:.2e}]"
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
        if ir_evaluator is not None:
            eval_results = ir_evaluator(model, epoch=epoch + 1)
            ndcg_key = "esci-eval_cosine_ndcg@10"
            mrr_key = "esci-eval_cosine_mrr@10"
            logger.info(
                "Eval       nDCG@10=%.4f  MRR@10=%.4f",
                eval_results.get(ndcg_key, 0.0),
                eval_results.get(mrr_key, 0.0),
            )
    if save_path is not None:  # If save path provided
        save_path = Path(save_path)  # Convert to Path object
        save_path.parent.mkdir(parents=True, exist_ok=True)  # Create parent directories if needed
        # Save model checkpoint: state dict and model name
        torch.save({"model_state": model.state_dict(), "model_name": model_name}, save_path)
    return model  # Return trained model


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
       - loss_scale: 1.0 (set to 20–50 for grad_norm in the 1–10 range like many contrastive setups).
       - reweight_hard: True.
       - hard_weight_power: 1.0.
       - use_query_augment: True.
       - augment_prob: 0.25.
       - save_path: `"data/model.pt"`.
       - use_saved_splits: False (load full ESCI from raw; set True or pass --use-saved-splits to use parquets).
    """
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Suppress verbose HTTP/redirect logs and HF unauthenticated warnings.
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    # Filter the common unauthenticated warning from huggingface_hub.
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
    args = p.parse_args()
    cfg: dict = {}
    config_path = REPO_ROOT / args.config
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    data_dir = Path(args.data_dir or cfg.get("data_dir") or DATA_DIR)
    # Call training function with args/config merged (CLI args override config)
    run_training(
        data_dir=data_dir,  # Data directory
        model_name=args.model_name or cfg.get("model_name", "all-MiniLM-L12-v2"),  # Model name
        shared_tower=args.shared or cfg.get("shared_tower", False),  # Shared tower flag
        batch_size=args.batch_size if args.batch_size is not None else cfg.get("batch_size", 64),  # Batch size
        epochs=args.epochs if args.epochs is not None else cfg.get("epochs", 3),  # Number of epochs
        lr=args.lr if args.lr is not None else cfg.get("lr", 2e-5),
        loss_scale=args.loss_scale if args.loss_scale is not None else cfg.get("loss_scale", 30.0),
        reweight_hard=not args.no_reweight_hard and cfg.get("reweight_hard", True),
        hard_weight_power=args.hard_weight_power if args.hard_weight_power is not None else cfg.get("hard_weight_power", 1.0),
        use_query_augment=not args.no_query_augment and cfg.get("use_query_augment", True),
        augment_prob=args.augment_prob if args.augment_prob is not None else cfg.get("augment_prob", 0.25),
        save_path=args.save or cfg.get("save_path", "data/model.pt"),
        use_saved_splits=args.use_saved_splits or cfg.get("use_saved_splits", False),
    )
    return 0


if __name__ == "__main__":
    main()
