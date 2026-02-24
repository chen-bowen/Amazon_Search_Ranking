# Amazon Search Retrieval

Two-tower (bi-encoder) query–product embedding model for product search, inspired by [Instacart ITEMS](https://tech.instacart.com/how-instacart-uses-embeddings-to-improve-search-relevance-e569839c3c36), trained on the [Amazon Shopping Queries (ESCI) dataset](https://github.com/amazon-science/esci-data).

## Setup

```bash
uv sync
```

## Data

Place the [Amazon ESCI](https://github.com/amazon-science/esci-data) parquet files in `data/` (or `data/esci-data/shopping_queries_dataset/`). Then load and save train/test splits:

```bash
uv run python -m src.data.load_esci --save-splits
```

This writes `data/esci_train.parquet` and `data/esci_test.parquet`.

## Train

```bash
uv run python -m src.training.train --config configs/train.yaml
```

## Evaluate

```bash
uv run python -m src.eval.evaluate --config configs/eval.yaml
```

## Retrieval

Build FAISS index and run retrieval:

```bash
uv run python -m src.retrieval.build_index --config configs/retrieval.yaml
uv run python -m src.retrieval.query --config configs/retrieval.yaml --query "organic milk"
```
