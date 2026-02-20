# Amazon Search Retrieval

Two-tower (bi-encoder) query–product embedding model for product search, inspired by [Instacart ITEMS](https://tech.instacart.com/how-instacart-uses-embeddings-to-improve-search-relevance-e569839c3c36), trained on the [Amazon Shopping Queries (ESCI) dataset](https://github.com/amazon-science/esci-data).

## Setup

```bash
uv sync
```

## Data

Download ESCI data (parquet files) into `data/` and run:

```bash
uv run python -m src.data.download_esci
uv run python -m src.data.prepare_esci
```

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
