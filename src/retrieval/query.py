"""
Query the FAISS index: embed query, run ANN, return top-k product IDs and scores.
Optional second stage: cross-encoder reranker (ESCI approach).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from src.constants import DEFAULT_MODEL_NAME, REPO_ROOT
from src.models.retriever import BiEncoderRetriever
from src.retrieval.build_index import load_index_and_meta

logger = logging.getLogger(__name__)


def search(
    query: str,
    model: BiEncoderRetriever,
    index,
    meta_df,
    *,
    top_k: int = 10,
    device: str | torch.device = "cuda",
) -> list[tuple[str, float]]:
    """
    Returns list of (product_id, score) for top-k. Score = cosine similarity (dot product on normalized).

    Parameters
    ----------
    - query : str
        Query string.
    - model : TwoTowerEncoder
        TwoTowerEncoder model.
    - index : faiss.Index
        FAISS index.
    - meta_df : pd.DataFrame
        Metadata DataFrame.
    - top_k : int
        Number of top products to return.
    - device : str | torch.device
        Device to use for encoding.

    Returns
    -------
    - out : list[tuple[str, float]]
        List of (product_id, score) for top-k.
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


def search_with_rerank(
    query: str,
    model: BiEncoderRetriever,
    index,
    meta_df,
    reranker,
    *,
    top_k: int = 10,
    rerank_top_k: int = 100,
    device: str | torch.device = "cuda",
) -> list[tuple[str, float]]:
    """
    Two-stage retrieval: (1) bi-encoder retrieves rerank_top_k candidates,
    (2) cross-encoder reranks them and returns top_k.

    Parameters
    ----------
    - query : str
        Query string.
    - model : TwoTowerEncoder
        Bi-encoder for stage 1.
    - index : faiss.Index
        FAISS index.
    - meta_df : pd.DataFrame
        Metadata with product_id and product_text.
    - reranker : Reranker
        Reranker for stage 2.
    - top_k : int
        Final number of products to return.
    - rerank_top_k : int
        Number of candidates to retrieve for reranking.
    - device : str | torch.device
        Device for bi-encoder.

    Returns
    -------
    - list[tuple[str, float]]
        List of (product_id, score) for top_k, scores from reranker.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    q_emb = model.encode_queries([query], device=device)
    q_np = q_emb.cpu().numpy().astype(np.float32)
    scores, indices = index.search(q_np, min(rerank_top_k, len(meta_df)))
    candidates = []
    for idx in indices[0]:
        if idx < 0:
            break
        row = meta_df.iloc[idx]
        pid = str(row["product_id"])
        text = str(row.get("product_text", ""))
        candidates.append((pid, text))
    if not candidates:
        return []
    return reranker.rerank(query, candidates, batch_size=32)[:top_k]


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
    p.add_argument("--no-rerank", action="store_true", help="Skip reranker even if configured")
    args = p.parse_args()
    cfg = {}
    config_path = REPO_ROOT / args.config
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiEncoderRetriever(
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
    top_k = args.top_k or cfg.get("top_k", 10)
    reranker_path = cfg.get("reranker_path")
    use_rerank = not args.no_rerank and reranker_path and Path(reranker_path).exists()
    if use_rerank:
        from src.models.reranker import load_reranker

        reranker = load_reranker(model_path=reranker_path, device=device)
        rerank_top_k = cfg.get("rerank_top_k", 100)
        results = search_with_rerank(
            args.query,
            model,
            index,
            meta,
            reranker,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
            device=device,
        )
    else:
        results = search(args.query, model, index, meta, top_k=top_k, device=device)
    for pid, score in results:
        logger.info("%.4f\t%s", score, pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
