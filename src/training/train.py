"""
Training loop: load relevant (query, product) pairs, in-batch negatives, self-adversarial loss.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

logger = logging.getLogger(__name__)
from torch.utils.data import Dataset  # Base class for datasets

from src.models.two_tower import TwoTowerEncoder  # Two-tower model
from src.training.loss import contrastive_loss_with_reweighting  # Contrastive loss with reweighting

REPO_ROOT = Path(__file__).resolve().parents[2]  # Go up 2 levels to project root
DATA_DIR = REPO_ROOT / "data"  # Path to data directory


class QueryProductDataset(Dataset):
    """Dataset of (query, product_text) pairs; positive pairs (relevance >= 2, i.e. E, S, C)."""

    def __init__(self, df: pd.DataFrame, min_relevance: int = 2):
        # Filter to positive pairs: relevance >= 2 (E=4, S=2, C=3; excludes I=1)
        # Keep only query and product_text columns, remove duplicates, reset index
        self.df = df[df["relevance"] >= min_relevance][["query", "product_text"]].drop_duplicates().reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)  # Return number of positive pairs

    def __getitem__(self, i: int) -> tuple[str, str]:
        row = self.df.iloc[i]  # Get row at index i
        return str(row["query"]), str(row["product_text"])  # Return (query string, product text string)


def collate_query_product(batch: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    """
    Collate function: convert list of (query, product) tuples into separate lists.
    """
    queries, products = zip(*batch)  # Unzip: [(q1,p1), (q2,p2), ...] -> ([q1,q2,...], [p1,p2,...])
    return list(queries), list(products)  # Convert tuples to lists


def run_training(
    train_df: pd.DataFrame | None = None,
    data_dir: Path | str | None = None,
    *,
    model_name: str = "all-MiniLM-L6-v2",
    shared_tower: bool = False,
    batch_size: int = 64,
    epochs: int = 3,
    lr: float = 2e-5,
    temperature: float = 0.05,
    reweight_hard: bool = True,
    hard_weight_power: float = 1.0,
    device: str | torch.device = "cuda",
    save_path: Path | str | None = None,
) -> TwoTowerEncoder:
    """
    Main training function: load data, create model, train with contrastive loss.
    """
    if train_df is None:  # If DataFrame not provided
        base = Path(data_dir or DATA_DIR)  # Use provided data_dir or default
        train_path = base / "esci_train.parquet"  # Path to preprocessed train parquet
        if not train_path.exists():  # If preprocessed file doesn't exist
            from src.data.load_data import load_esci, prepare_train_test
            # Load raw ESCI data and split into train/test
            _df = load_esci(data_dir=base / "esci-data" / "shopping_queries_dataset")
            train_df, _ = prepare_train_test(df=_df)  # Extract train split
        else:  # If preprocessed file exists
            train_df = pd.read_parquet(train_path)  # Load preprocessed train data
    device = torch.device(device if torch.cuda.is_available() else "cpu")  # Use GPU if available, else CPU
    model = TwoTowerEncoder(model_name=model_name, shared=shared_tower, normalize=True)  # Create two-tower model
    model = model.to(device)  # Move model to device (GPU/CPU)
    dataset = QueryProductDataset(train_df)  # Create dataset from train DataFrame
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,  # Dataset to iterate over
        batch_size=batch_size,  # Number of samples per batch
        shuffle=True,  # Shuffle batches each epoch
        collate_fn=collate_query_product,  # Function to combine samples into batch
        num_workers=0,  # No multiprocessing (0 = single-threaded)
        drop_last=True,  # Drop last incomplete batch
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr)  # Create optimizer (AdamW with learning rate)
    model.train()  # Set model to training mode (enables dropout, batch norm updates, etc.)
    for epoch in range(epochs):  # Loop over epochs
        total_loss = 0.0  # Accumulator for epoch loss
        n_batches = 0  # Counter for batches
        for queries, products in dataloader:  # Iterate over batches
            opt.zero_grad()  # Zero gradients from previous iteration
            # Forward pass: encode queries and products, get embeddings
            q_emb, p_emb = model(
                query_strings=queries,  # List of query strings
                product_strings=products,  # List of product text strings
                device=device,  # Device to run on
            )
            # Compute contrastive loss with in-batch negatives and self-adversarial reweighting
            loss = contrastive_loss_with_reweighting(
                q_emb, p_emb,  # Query and product embeddings
                temperature=temperature,  # Temperature for softmax scaling
                reweight_hard=reweight_hard,  # Whether to reweight hard negatives
                hard_weight_power=hard_weight_power,  # Power for reweighting (higher = more focus on hard negatives)
            )
            loss.backward()  # Backward pass: compute gradients
            opt.step()  # Update model parameters using gradients
            total_loss += loss.item()  # Add batch loss to accumulator (detach from graph)
            n_batches += 1  # Increment batch counter
        logger.info("Epoch %s/%s loss=%.4f", epoch + 1, epochs, total_loss / max(n_batches, 1))
    if save_path is not None:  # If save path provided
        save_path = Path(save_path)  # Convert to Path object
        save_path.parent.mkdir(parents=True, exist_ok=True)  # Create parent directories if needed
        # Save model checkpoint: state dict and model name
        torch.save({"model_state": model.state_dict(), "model_name": model_name}, save_path)
    return model  # Return trained model


def main() -> int:
    """Command-line entry point: parse args, load config, run training."""
    import yaml
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Train two-tower encoder on ESCI")  # Create argument parser
    p.add_argument("--config", type=str, default="configs/train.yaml", help="Config YAML (optional)")  # Config file path
    p.add_argument("--data-dir", type=str, default=None)  # Override data directory
    p.add_argument("--model-name", type=str, default=None)  # Override model name
    p.add_argument("--shared", action="store_true")  # Use shared encoder for both towers
    p.add_argument("--batch-size", type=int, default=None)  # Override batch size
    p.add_argument("--epochs", type=int, default=None)  # Override number of epochs
    p.add_argument("--lr", type=float, default=None)  # Override learning rate
    p.add_argument("--temperature", type=float, default=None)  # Override temperature
    p.add_argument("--no-reweight-hard", action="store_true", help="Disable self-adversarial reweighting")  # Disable reweighting
    p.add_argument("--hard-weight-power", type=float, default=None)  # Override hard weight power
    p.add_argument("--save", type=str, default=None)  # Override save path
    args = p.parse_args()  # Parse command-line arguments
    cfg = {}  # Initialize config dict
    config_path = REPO_ROOT / args.config  # Full path to config file
    if config_path.exists():  # If config file exists
        with open(config_path) as f:  # Open config file
            cfg = yaml.safe_load(f) or {}  # Load YAML into dict (empty dict if None)
    data_dir = Path(args.data_dir or cfg.get("data_dir") or DATA_DIR)  # Use arg > config > default
    # Call training function with args/config merged (CLI args override config)
    run_training(
        data_dir=data_dir,  # Data directory
        model_name=args.model_name or cfg.get("model_name", "all-MiniLM-L6-v2"),  # Model name
        shared_tower=args.shared or cfg.get("shared_tower", False),  # Shared tower flag
        batch_size=args.batch_size if args.batch_size is not None else cfg.get("batch_size", 64),  # Batch size
        epochs=args.epochs if args.epochs is not None else cfg.get("epochs", 3),  # Number of epochs
        lr=args.lr if args.lr is not None else cfg.get("lr", 2e-5),  # Learning rate
        temperature=args.temperature if args.temperature is not None else cfg.get("temperature", 0.05),  # Temperature
        reweight_hard=not args.no_reweight_hard and cfg.get("reweight_hard", True),  # Reweighting enabled
        hard_weight_power=args.hard_weight_power if args.hard_weight_power is not None else cfg.get("hard_weight_power", 1.0),  # Hard weight power
        save_path=args.save or cfg.get("save_path", "data/model.pt"),  # Save path
    )
    return 0  # Exit successfully


if __name__ == "__main__":
    raise SystemExit(main())  # Run main and exit with return code
