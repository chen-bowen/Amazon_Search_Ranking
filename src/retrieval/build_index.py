"""
Build FAISS index from product embeddings and save product_id / product_text metadata.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.constants import DATA_DIR, DEFAULT_MODEL_NAME
from src.data.load_data import load_esci
from src.models.two_tower import TwoTowerEncoder

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    faiss = None


def build_faiss_index(
    model: TwoTowerEncoder,
    products_df: pd.DataFrame,
    *,
    product_id_col: str = "product_id",
    product_text_col: str = "product_text",
    index_path: Path | str | None = None,
    meta_path: Path | str | None = None,
    device: str | torch.device = "cuda",
    batch_size: int = 128,
) -> tuple[faiss.Index, pd.DataFrame]:
    """
    Encode all product_text, build FAISS index (inner product on normalized = cosine).
    products_df should have product_id and product_text (expanded). Returns (index, meta_df).
    """
    if faiss is None:
        raise ImportError("faiss-cpu or faiss-gpu is required")
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    texts = products_df[product_text_col].astype(str).tolist()
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        e = model.encode_products(batch, device=device)
        embs.append(e.cpu().numpy())
    embs = np.vstack(embs).astype(np.float32)
    # FAISS index for inner product (IndexFlatIP) with normalized vectors = cosine
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    meta = products_df[[product_id_col, product_text_col]].copy()
    if index_path:
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
    if meta_path:
        Path(meta_path).parent.mkdir(parents=True, exist_ok=True)
        meta.to_parquet(meta_path, index=False)
    return index, meta


def load_index_and_meta(
    index_path: Path | str,
    meta_path: Path | str,
) -> tuple[faiss.Index, pd.DataFrame]:
    if faiss is None:
        raise ImportError("faiss-cpu or faiss-gpu is required")
    index = faiss.read_index(str(index_path))
    meta = pd.read_parquet(meta_path)
    return index, meta


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Build FAISS index from product embeddings")
    p.add_argument("--config", type=str, default="configs/retrieval.yaml")
    p.add_argument("--model-path", type=str, default="data/model.pt")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument(
        "--products",
        type=str,
        default=None,
        help="Parquet with product_id, product_text",
    )
    p.add_argument("--index", type=str, default="data/product.index")
    p.add_argument("--meta", type=str, default="data/product_meta.parquet")
    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoTowerEncoder(model_name=args.model_name, shared=False, normalize=True)
    if Path(args.model_path).exists():
        ckpt = torch.load(args.model_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"], strict=True)
    if args.products and Path(args.products).exists():
        products_df = pd.read_parquet(args.products)
    else:
        base = Path(args.data_dir or DATA_DIR)
        # Load ESCI parquets from base data directory (expects files directly under `data/`).
        df = load_esci(data_dir=base)
        products_df = (
            df[["product_id", "product_text"]]
            .drop_duplicates("product_id")
            .reset_index(drop=True)
        )
    build_faiss_index(
        model, products_df, index_path=args.index, meta_path=args.meta, device=device
    )
    logger.info("Index saved: %s, meta: %s", args.index, args.meta)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
