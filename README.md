## Amazon Search Retrieval

Two-tower (bi-encoder) query–product model for Amazon product search, inspired by [Instacart ITEMS](https://tech.instacart.com/how-instacart-uses-embeddings-to-improve-search-relevance-e569839c3c36) and trained on the [Amazon Shopping Queries (ESCI) dataset](https://github.com/amazon-science/esci-data).

The model encodes:

- **Query tower**: user query text (optionally augmented).
- **Product tower**: product text (title, bullet points, etc.).

Training uses a contrastive loss with in-batch negatives; at retrieval time we encode the query once and rank products by cosine similarity.

---

## Setup

### Python environment

```bash
uv sync
```

This creates a virtualenv (if needed) and installs all dependencies from `pyproject.toml` / `uv.lock`.

---

## Data

### Input

Download the [Amazon ESCI](https://github.com/amazon-science/esci-data) parquet files into `data/` (or `data/esci-data/shopping_queries_dataset/`), e.g.:

- `shopping_queries_dataset_products.parquet`
- `shopping_queries_dataset_examples.parquet`

Then materialize train / eval splits:

```bash
uv run python -m src.data.load_esci --save-splits
```

This writes:

- `data/esci_train.parquet`
- `data/esci_test.parquet`

These are what the training script reads by default.

---

## Training

### Config-driven training

Main entrypoint:

```bash
uv run python -m src.training.train --config configs/train.yaml
```

`configs/train.yaml` controls:

- `data_dir`: where `esci_train.parquet` / `esci_test.parquet` live.
- `model_name`: SentenceTransformer backbone (e.g. `all-MiniLM-L12-v2`).
- `batch_size`, `epochs`, `lr`, `temperature`, `reweight_hard`, etc.
- `save_path`: where to write the final checkpoint.

CLI flags override config values (see `main()` in `src/training/train.py`).

### What the training loop does

In `run_training` (`src/training/train.py`):

- Builds a `QueryProductDataset` of positive `(query, product_text)` pairs from `esci_train.parquet`.
- Uses a standard `DataLoader` with `collate_query_product` to get `List[str]` queries and products.
- Feeds them into `TwoTowerEncoder` (`src/models/two_tower.py`) to get L2‑normalized embeddings.
- Applies `contrastive_loss_with_reweighting` (`src/training/loss.py`) with optional hard‑negative reweighting.
- Optimizes with AdamW + warmup + cosine decay.

Progress display:

- **Global tqdm bar** over _all steps in all epochs_:
  - Shows overall progress and ETA for the full run.
- **Second “stats” line** under the bar:
  - Displays `Epoch e/E | loss, last50, |g|, lr`.

This mirrors the UX from the Instacart project, but adapted for this ESCI setup.

---

## Information Retrieval evaluation

We use `sentence_transformers.evaluation.InformationRetrievalEvaluator` for IR metrics on `esci_test.parquet`.

### Eval artifacts

`build_ir_eval_data` in `src/training/train.py` converts the eval DataFrame into:

- `queries`: `query_id -> query text`
- `corpus`: `product_id -> product_text`
- `relevant_docs`: `query_id -> set[product_id]` with relevance ≥ 2 (E/S/C)

### Two evaluators: subsample + full

To keep eval tractable on a laptop, `build_ir_evaluators` creates:

- **Mid‑epoch evaluator (subsample)**:
  - At most `max_mid_queries` (default 2000) queries.
  - At most `max_mid_corpus` (default 50k) corpus docs.
  - Name: `esci-eval-subsample`.
- **End‑of‑epoch evaluator (full)**:
  - All eval queries and all corpus docs.
  - Name: `esci-eval`.

Both are configured with:

- `mrr_at_k=[10]`
- `ndcg_at_k=[10]`
- `accuracy_at_k=[1, 10]`
- `precision_recall_at_k=[10]`
- `map_at_k=[10, 100]`

### When eval runs

Inside the training loop:

- **Mid‑epoch** (on subsample): every ~⅓ of an epoch (configurable):
  - Uses `ir_evaluator_mid`.
  - Logs: `nDCG@10`, `MRR@10`, `Acc@10`, `Recall@10`, `MAP@10`.
- **End‑of‑epoch** (on full eval set):
  - Uses `ir_evaluator_full`.
  - Logs the same metrics as above.

We also:

- Use `tqdm.write` / `progress.write` so eval messages live on their own line.
- Silence the verbose `InformationRetrievalEvaluator` logger via `setup_colored_logging` in `src/utils.py`.

---

## Retrieval

After training a checkpoint, you can build an index and run retrieval over the ESCI products.

### Build FAISS index

```bash
uv run python -m src.retrieval.build_index --config configs/retrieval.yaml
```

This:

- Loads the trained two‑tower model.
- Encodes all products into embeddings.
- Builds and persists a FAISS index for fast ANN search.

### Run queries

```bash
uv run python -m src.retrieval.query --config configs/retrieval.yaml --query "organic milk"
```

This:

- Encodes the input query string.
- Searches the FAISS index.
- Prints the top‑k product IDs and their scores.

---

## Second-stage reranker (ESCI approach)

A cross-encoder reranker improves ranking by scoring (query, product) pairs jointly. This follows the [ESCI baseline](https://github.com/amazon-science/esci-data) (Task 1): MSE loss on ESCI gains (E=1.0, S=0.1, C=0.01, I=0.0).

### Train the reranker

```bash
uv run python -m src.training.train_reranker --config configs/reranker.yaml
```

This trains `cross-encoder/ms-marco-MiniLM-L-12-v2` on ESCI and saves to `data/reranker`.

### Two-stage retrieval

When `reranker_path` is set in `configs/retrieval.yaml` and the path exists, queries use:

1. **Stage 1**: Bi-encoder retrieves `rerank_top_k` candidates (default 100).
2. **Stage 2**: Cross-encoder reranks them and returns `top_k` (default 10).

To skip the reranker even when configured:

```bash
uv run python -m src.retrieval.query --query "organic milk" --no-rerank
```

---

## Notes on devices and performance

- `resolve_device` prefers CUDA, then Apple `mps`, then CPU.
- On Apple Silicon (`mps`):
  - `dataloader_num_workers=0` to avoid multiprocessing issues.
  - No fp16; we keep everything in fp32 for stability.
  - Expect slower steps than a good CUDA GPU, but the subsampled IR eval keeps feedback reasonable.

If training is too slow, you can:

- Reduce `epochs` or `batch_size` in `configs/train.yaml`.
- Disable IR eval entirely by skipping evaluators in `run_training` (or adding a config/flag).
- Further tighten the subsample (`max_mid_queries`, `max_mid_corpus`) for mid‑epoch evals.

---

## Project structure

Key files:

- `src/training/train.py`: end‑to‑end training loop and IR evaluation wiring.
- `src/training/train_reranker.py`: cross-encoder reranker training (ESCI approach).
- `src/models/two_tower.py`: query/product encoders with a small shim for `InformationRetrievalEvaluator`.
- `src/models/reranker.py`: cross-encoder load/rerank utilities.
- `src/training/loss.py`: contrastive loss with optional hard‑negative reweighting.
- `src/retrieval/build_index.py`: build and save FAISS index from product embeddings.
- `src/retrieval/query.py`: CLI for querying (with optional two-stage reranking).
- `configs/train.yaml`: default training config.
- `configs/reranker.yaml`: reranker training config.
- `configs/retrieval.yaml`: retrieval/index config.

The overall design closely follows the Instacart two‑tower project ([`instacart_next_order_recommendation`](https://github.com/chen-bowen/instacart_next_order_recommendation)), adapted to Amazon ESCI and query–product search.
