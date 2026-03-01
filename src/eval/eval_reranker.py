#!/usr/bin/env python3
"""
Evaluate trained ESCI reranker: compute nDCG, MRR, MAP, Recall@10 on test set.
Run: uv run python scripts/eval_reranker.py [--model-path data/reranker]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import DATA_DIR
from src.data.load_data import load_esci, prepare_train_test
from src.models.reranker import load_reranker
from src.eval.esci_evaluator import ESCIMetricsEvaluator


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate ESCI reranker (nDCG, MRR, MAP, Recall@10)")
    p.add_argument("--model-path", type=str, default="data/reranker", help="Path to saved reranker")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--product-col", type=str, default="product_text")
    p.add_argument("--max-queries", type=int, default=None, help="Subsample for faster eval")
    p.add_argument("--recall-at", type=int, default=10, help="Recall@k (default 10)")
    p.add_argument("--small-version", action="store_true", help="Use Task 1 reduced set")
    args = p.parse_args()

    base = Path(args.data_dir or DATA_DIR)
    test_path = base / "esci_test.parquet"
    if test_path.exists():
        import pandas as pd

        test_df = pd.read_parquet(test_path)
    else:
        df = load_esci(data_dir=base, small_version=args.small_version)
        _, test_df = prepare_train_test(df=df)

    if len(test_df) == 0:
        print("No test data found.")
        return 1

    reranker = load_reranker(model_path=args.model_path)
    evaluator = ESCIMetricsEvaluator(
        test_df,
        product_col=args.product_col,
        max_queries=args.max_queries,
        batch_size=32,
        recall_at_k=args.recall_at,
    )
    evaluator(reranker, output_path=None, epoch=-1, steps=-1)
    metrics = evaluator._last_metrics
    print(f"nDCG = {metrics['ndcg']:.4f}")
    print(f"MRR  = {metrics['mrr']:.4f}")
    print(f"MAP  = {metrics['map']:.4f}")
    print(f"Recall@{args.recall_at} = {metrics['recall']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
