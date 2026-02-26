"""
Query the FAISS index: embed query, run ANN, return top-k product IDs and scores.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from src.constants import DEFAULT_MODEL_NAME, REPO_ROOT
from src.models.two_tower import TwoTowerEncoder
from src.retrieval.build_index import load_index_and_meta

logger = logging.getLogger(__name__)


def search(
    query: str,
    model: TwoTowerEncoder,
    index,
    meta_df,
    *,
    top_k: int = 10,
    device: str | torch.device = "cuda",
) -> list[tuple[str, float]]:
    """
    Returns list of (product_id, score) for top-k. Score = cosine similarity (dot product on normalized).
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    q_emb = model.encode_queries([query], device=device)
    q_np = q_emb.cpu().numpy().astype(np.float32)
    scores, indices = index.search(q_np, min(top_k, len(meta_df)))
    out = []
    for idx, sc in zip(indices[0], scores[0]):
        if idx < 0:
            break
        pid = meta_df.iloc[idx]["product_id"]
        out.append((str(pid), float(sc)))
    return out


def main() -> int:
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Query FAISS index with a search string")
    p.add_argument("--config", type=str, default="configs/retrieval.yaml")
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--index", type=str, default=None)
    p.add_argument("--meta", type=str, default=None)
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--top-k", type=int, default=None)
    args = p.parse_args()
    cfg = {}
    config_path = REPO_ROOT / args.config
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoTowerEncoder(
        model_name=args.model_name or cfg.get("model_name", DEFAULT_MODEL_NAME),
        shared=False,
        normalize=True,
    )
    model_path = args.model_path or cfg.get("model_path", "data/model.pt")
    if Path(model_path).exists():
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"], strict=True)
    index_path = args.index or cfg.get("index_path", "data/product.index")
    meta_path = args.meta or cfg.get("meta_path", "data/product_meta.parquet")
    index, meta = load_index_and_meta(index_path, meta_path)
    results = search(
        args.query,
        model,
        index,
        meta,
        top_k=args.top_k or cfg.get("top_k", 10),
        device=device,
    )
    for pid, score in results:
        logger.info("%.4f\t%s", score, pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
