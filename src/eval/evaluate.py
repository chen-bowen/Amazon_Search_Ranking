"""
Evaluate two-tower model on ESCI test set: compute nDCG@k and MRR.
"""
from __future__ import annotations  # Enable postponed evaluation of type hints

import argparse  # For command-line argument parsing
from pathlib import Path  # For path manipulation

import numpy as np  # For array operations
import pandas as pd  # For DataFrame operations
import torch  # PyTorch tensor operations

from src.eval.metrics import evaluate_ranking  # Ranking evaluation functions
from src.models.two_tower import TwoTowerEncoder  # Two-tower model

REPO_ROOT = Path(__file__).resolve().parents[2]  # Go up 2 levels to project root
DATA_DIR = REPO_ROOT / "data"  # Path to data directory
BATCH_SIZE = 128  # Batch size for encoding (not currently used, but available)


def run_evaluation(
    model: TwoTowerEncoder | None = None,
    model_path: str | Path | None = None,
    test_df: pd.DataFrame | None = None,
    data_dir: Path | str | None = None,
    *,
    model_name: str = "all-MiniLM-L6-v2",
    k: int = 10,
    device: str | torch.device = "cuda",
) -> dict[str, float]:
    """
    Evaluate model on test set: compute embeddings, rank products, compute nDCG@k and MRR.
    """
    if test_df is None:  # If test DataFrame not provided
        base = Path(data_dir or DATA_DIR)  # Use provided data_dir or default
        test_path = base / "esci_test.parquet"  # Path to preprocessed test parquet
        if not test_path.exists():  # If preprocessed file doesn't exist
            from src.data.load_esci import load_esci, prepare_train_test
            # Load raw ESCI data and split into train/test
            _, test_df = prepare_train_test(data_dir=base / "esci-data" / "shopping_queries_dataset")
        else:  # If preprocessed file exists
            test_df = pd.read_parquet(test_path)  # Load preprocessed test data
    device = torch.device(device if torch.cuda.is_available() else "cpu")  # Use GPU if available, else CPU
    if model is None:  # If model not provided
        model = TwoTowerEncoder(model_name=model_name, shared=False, normalize=True)  # Create new model
        if model_path is not None:  # If model checkpoint path provided
            ckpt = torch.load(Path(model_path), map_location=device, weights_only=True)  # Load checkpoint
            model.load_state_dict(ckpt["model_state"], strict=True)  # Load model weights
        model = model.to(device)  # Move model to device (GPU/CPU)
    model.eval()  # Set model to evaluation mode (disables dropout, batch norm uses running stats)
    test_df = test_df.copy()  # Copy DataFrame to avoid modifying original
    test_df["_row_idx"] = np.arange(len(test_df))  # Add row index column to track original order
    scores_arr = np.full(len(test_df), np.nan, dtype=float)  # Initialize scores array with NaN
    with torch.no_grad():  # Disable gradient computation (faster, less memory)
        for _qid, grp in test_df.groupby("query_id"):  # Group by query_id (iterate over each query)
            query = grp["query"].iloc[0]  # Get query string (same for all rows in group)
            products = grp["product_text"].tolist()  # Get list of product texts for this query
            row_indices = grp["_row_idx"].values  # Get original row indices for this group
            # Encode query (repeat for each product) and products
            q_embs = model.encode_queries([query] * len(products), device=device)  # [N, D] query embeddings
            p_embs = model.encode_products(products, device=device)  # [N, D] product embeddings
            # Compute similarity: dot product (cosine for normalized embeddings)
            sim = (q_embs * p_embs).sum(dim=1).cpu().numpy()  # [N] similarity scores, move to CPU
            scores_arr[row_indices] = sim  # Store scores at original row positions
    test_df.drop(columns=["_row_idx"], inplace=True)  # Remove temporary row index column
    # Evaluate ranking: compute nDCG@k and MRR per query, return averages
    return evaluate_ranking(test_df, scores_arr, query_id_col="query_id", relevance_col="relevance", k=k)


def main() -> int:
    """
    Command-line entry point: parse args, load config, run evaluation.
    """
    import yaml  # For YAML config file parsing
    p = argparse.ArgumentParser(description="Evaluate two-tower on ESCI test set")  # Create argument parser
    p.add_argument("--config", type=str, default="configs/eval.yaml")  # Config file path
    p.add_argument("--model-path", type=str, default=None)  # Override model checkpoint path
    p.add_argument("--data-dir", type=str, default=None)  # Override data directory
    p.add_argument("--model-name", type=str, default=None)  # Override model name
    p.add_argument("--k", type=int, default=None)  # Override k for nDCG@k
    args = p.parse_args()  # Parse command-line arguments
    cfg = {}  # Initialize config dict
    config_path = REPO_ROOT / args.config  # Full path to config file
    if config_path.exists():  # If config file exists
        with open(config_path) as f:  # Open config file
            cfg = yaml.safe_load(f) or {}  # Load YAML into dict (empty dict if None)
    data_dir = Path(args.data_dir or cfg.get("data_dir") or DATA_DIR)  # Use arg > config > default
    # Call evaluation function with args/config merged (CLI args override config)
    metrics = run_evaluation(
        model_path=args.model_path or cfg.get("model_path", "data/model.pt"),  # Model checkpoint path
        data_dir=data_dir,  # Data directory
        model_name=args.model_name or cfg.get("model_name", "all-MiniLM-L6-v2"),  # Model name
        k=args.k if args.k is not None else cfg.get("k", 10),  # k for nDCG@k
    )
    print("Metrics:", metrics)  # Print evaluation metrics
    return 0  # Exit successfully


if __name__ == "__main__":
    raise SystemExit(main())  # Run main and exit with return code
