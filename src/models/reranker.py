"""
Cross-encoder reranker for (query, product) pairs (ESCI-style).
Scores pairs jointly for ESCI Task 1 (query-product ranking).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from sentence_transformers.cross_encoder import CrossEncoder

from src.constants import DEFAULT_RERANKER_MODEL, MODEL_CACHE_DIR


class CrossEncoderReranker(nn.Module):
    """
    Cross-encoder that scores (query, product) pairs jointly.

    Notation used in comments:
    - B: batch size (number of pairs to score).
    - Output: scalar relevance score per pair.

    Can score any (query, product) candidate list for ranking.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        max_length: int = 512,
        device: str | torch.device | None = None,
        cache_folder: str | Path | None = MODEL_CACHE_DIR,
    ):
        super().__init__()
        if CrossEncoder is None:
            raise ImportError("sentence-transformers is required for Reranker")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        cache = str(cache_folder) if cache_folder else None
        self._model = CrossEncoder(
            model_name,
            num_labels=1,
            max_length=max_length,
            device=str(device),
            cache_folder=cache,
        )

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        device: str | torch.device | None = None,
    ) -> "CrossEncoderReranker":
        """
        Load from saved checkpoint (directory with config, model weights).

        Parameters
        ----------
        path : str | Path
            Directory containing saved CrossEncoder.
        device : str | torch.device | None
            Device to load on.

        Returns
        -------
        Reranker
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Reranker not found: {path}")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self._device = device
        self._model = CrossEncoder(str(path), device=str(device), local_files_only=True)
        return self

    def save(self, path: str | Path) -> None:
        """Save model to directory."""
        self._model.save(str(path))

    def predict(
        self,
        pairs: List[List[str]],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> List[float]:
        """
        Score (query, product) pairs.

        Parameters
        ----------
        pairs : List[List[str]]
            List of [query, product_text] pairs.
        batch_size : int
            Batch size for scoring.
        show_progress_bar : bool
            Whether to show progress.

        Returns
        -------
        List[float]
            Relevance score per pair.
        """
        if not pairs:
            return []
        scores = self._model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )
        if hasattr(scores, "tolist"):
            return scores.tolist()
        return [float(s) for s in scores]

    def rerank(
        self,
        query: str,
        candidates: List[tuple[str, str]],
        *,
        batch_size: int = 32,
    ) -> List[tuple[str, float]]:
        """
        Rerank (product_id, product_text) candidates for a single query.

        Parameters
        ----------
        query : str
            Query string.
        candidates : List[tuple[str, str]]
            List of (product_id, product_text) pairs to score.
        batch_size : int
            Batch size for scoring.

        Returns
        -------
        List[tuple[str, float]]
            List of (product_id, score) sorted by score descending.
        """
        if not candidates:
            return []
        pairs = [[query, text] for _pid, text in candidates]
        scores = self.predict(pairs, batch_size=batch_size)
        out = [(pid, float(sc)) for (pid, _), sc in zip(candidates, scores)]
        out.sort(key=lambda x: x[1], reverse=True)
        return out


def load_reranker(
    model_path: str | Path | None = None,
    model_name: str = DEFAULT_RERANKER_MODEL,
    device: str | torch.device | None = None,
    cache_folder: str | Path | None = MODEL_CACHE_DIR,
) -> CrossEncoderReranker:
    """
    Load a reranker. If model_path exists, load from disk; else use pretrained model_name.

    Parameters
    ----------
    model_path : str | Path | None
        Path to saved model directory. If exists, loads from here.
    model_name : str
        HuggingFace model id when model_path is not used.
    device : str | torch.device | None
        Device to load model on.

    Returns
    -------
    Reranker
    """
    path = Path(model_path) if model_path else None
    if path and path.exists():
        return CrossEncoderReranker.from_pretrained(path, device=device)
    return CrossEncoderReranker(model_name=model_name, device=device, cache_folder=cache_folder)
