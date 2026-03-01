## Amazon Search – ESCI Reranker

Cross-encoder reranker for [Amazon ESCI Task 1](https://github.com/amazon-science/esci-data) (query-product ranking). Trained on the [Shopping Queries Dataset](https://arxiv.org/abs/2206.06588) with MSE loss on ESCI gains (E=1.0, S=0.1, C=0.01, I=0.0). Target: nDCG ~0.852 (paper baseline for US).

---

## Setup

```bash
uv sync
```

---

## Data

Download the [Amazon ESCI](https://github.com/amazon-science/esci-data) parquets into `data/`:

- `shopping_queries_dataset_products.parquet`
- `shopping_queries_dataset_examples.parquet`

Materialize train/test splits:

```bash
uv run python -m src.data.load_data --save-splits
```

This writes `data/esci_train.parquet` and `data/esci_test.parquet`.

---

## Training

```bash
uv run python -m src.training.train_reranker --config configs/reranker.yaml
```

`configs/reranker.yaml` controls:

- `model_name`: Cross-encoder (default `cross-encoder/ms-marco-MiniLM-L-12-v2`)
- `product_col`: `product_text` (full) or `product_title` (ESCI-exact)
- `batch_size`, `epochs`, `lr`, `warmup_steps`
- `evaluation_steps`: run nDCG eval every N steps (5000)
- `save_path`: where to save (`data/reranker`)

CLI flags override config (e.g. `--epochs 2`, `--eval-max-queries 1000` for faster eval).

---

## Evaluation

```bash
uv run python scripts/eval_reranker.py --model-path data/reranker
```

Prints nDCG on the test set. Use `--max-queries 1000` for a quick subsample eval.

---

## Project structure

- `src/training/train_reranker.py` – reranker training + nDCG evaluator
- `src/models/reranker.py` – `CrossEncoderReranker`, `load_reranker`
- `scripts/eval_reranker.py` – standalone nDCG evaluation
- `configs/reranker.yaml` – main training config
- `notebooks/train_esci_reranker.ipynb` – training notebook
- `notebooks/inference_esci_reranker.ipynb` – inference notebook
- `notebooks/load_esci_data.ipynb` – data loading
