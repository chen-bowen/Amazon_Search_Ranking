"""
Multi-task learning cross-encoder for ESCI: Task 1 (query-product ranking),
Task 2 (4-class E/S/C/I classification), Task 3 (substitute identification).
Shared encoder with three heads.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.constants import (
    DEFAULT_RERANKER_MODEL,
    ESCI_ID2LABEL,
    MODEL_CACHE_DIR,
)
from src.utils import resolve_device

logger = logging.getLogger(__name__)


def _load_encoder_and_tokenizer(
    path: Path,
) -> tuple[AutoModel, AutoTokenizer, int, int]:
    """
    Load encoder and tokenizer from a local checkpoint directory.

    Returns encoder, tokenizer, hidden_size, and max_length.
    """
    encoder = AutoModel.from_pretrained(str(path), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    hidden_size = encoder.config.hidden_size
    max_length = getattr(
        tokenizer,
        "model_max_length",
        getattr(encoder.config, "max_position_embeddings", 512),
    )
    return encoder, tokenizer, hidden_size, max_length


def _load_heads_from_checkpoint(
    path: Path,
    device: torch.device,
    hidden_size: int,
) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
    """
    Build and, if available, load multi-task heads from checkpoint.

    Heads are initialized randomly when no checkpoint file is present.
    """
    head_ranking = nn.Linear(hidden_size, 1)
    head_esci = nn.Linear(hidden_size, 4)
    head_substitute = nn.Linear(hidden_size, 1)

    heads_path = path / "multi_task_heads.pt"
    if heads_path.exists():
        state = torch.load(heads_path, map_location=device, weights_only=True)
        head_ranking.load_state_dict(state["ranking"])
        head_esci.load_state_dict(state["esci"])
        head_substitute.load_state_dict(state["substitute"])
    else:
        logger.warning("No multi_task_heads.pt found at %s; head weights are random.", path)

    return head_ranking, head_esci, head_substitute


class MultiTaskReranker(nn.Module):
    """
    Cross-encoder with one shared backbone and three heads for ESCI tasks.

    - Task 1 (query-product ranking): scalar score for (query, product);
      MSE to ESCI gain.
    - Task 2 (4-class E/S/C/I): ESCI class logits; CrossEntropy.
    - Task 3 (substitute identification): binary logit (S vs non-S); BCE.

    Notation: B = batch size, H = hidden_size.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        max_length: int = 512,
        device: str | torch.device | None = None,
        cache_folder: str | Path | None = MODEL_CACHE_DIR,
    ) -> None:
        """
        Build multi-task learning model from a pretrained encoder name.

        Parameters
        ----------
        model_name : str
            HuggingFace model id (e.g. cross-encoder/ms-marco-MiniLM-L-12-v2).
        max_length : int
            Max token length for [query, product] concatenation.
        device : str | torch.device | None
            Device to place the model on.
        cache_folder : str | Path | None
            Directory for downloading/caching pretrained weights.
        """
        super().__init__()
        self._device = resolve_device(device)
        self._max_length = max_length
        cache = str(cache_folder) if cache_folder else None

        config = AutoConfig.from_pretrained(model_name, cache_dir=cache)
        self._hidden_size = config.hidden_size

        # Shared encoder (no classification head).
        self.encoder = AutoModel.from_pretrained(model_name, config=config, cache_dir=cache)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache)
        self.tokenizer.model_max_length = min(
            getattr(self.tokenizer, "model_max_length", 512),
            getattr(config, "max_position_embeddings", 512),
        )

        # Multi-task learning Task 1 (query-product ranking): regression head.
        self.head_ranking = nn.Linear(self._hidden_size, 1)
        # Multi-task learning Task 2 (4-class E/S/C/I): classification head.
        self.head_esci = nn.Linear(self._hidden_size, 4)
        # Multi-task learning Task 3 (substitute identification): binary head.
        self.head_substitute = nn.Linear(self._hidden_size, 1)

        self.to(self.device)

    @property
    def device(self) -> torch.device:
        """Device the model is on (for external callers)."""
        return self._device

    @property
    def max_length(self) -> int:
        """Max token length for inputs."""
        return self._max_length

    @property
    def hidden_size(self) -> int:
        """Encoder hidden size."""
        return self._hidden_size

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        device: str | torch.device | None = None,
    ) -> MultiTaskReranker:
        """
        Load multi-task learning reranker from a saved checkpoint directory.

        The directory must contain encoder and tokenizer (e.g. config.json,
        model.safetensors, tokenizer files) plus multi_task_heads.pt for the
        three heads.

        Parameters
        ----------
        path : str | Path
            Directory containing the saved multi-task learning checkpoint.
        device : str | torch.device | None
            Device to load the model on.

        Returns
        -------
        MultiTaskReranker
            Loaded model.

        Raises
        ------
        FileNotFoundError
            If path does not exist or required files are missing.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Multi-task learning reranker not found: {path}")

        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self._device = resolve_device(device)

        (
            self.encoder,
            self.tokenizer,
            self._hidden_size,
            self._max_length,
        ) = _load_encoder_and_tokenizer(path)
        (
            self.head_ranking,
            self.head_esci,
            self.head_substitute,
        ) = _load_heads_from_checkpoint(path, self.device, self._hidden_size)

        self.to(self.device)
        return self

    def save(self, path: str | Path) -> None:
        """
        Save encoder, tokenizer, and multi-task learning heads to a directory.

        Parameters
        ----------
        path : str | Path
            Directory to write to. Created if it does not exist.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        torch.save(
            {
                "ranking": self.head_ranking.state_dict(),
                "esci": self.head_esci.state_dict(),
                "substitute": self.head_substitute.state_dict(),
            },
            path / "multi_task_heads.pt",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run shared encoder and three heads.

        Parameters
        ----------
        input_ids : torch.Tensor
            (B, L) token ids.
        attention_mask : torch.Tensor
            (B, L) attention mask.
        token_type_ids : torch.Tensor | None
            Optional (B, L) segment ids.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (scores, esci_logits, substitute_logits).
            Shapes: (B,), (B, 4), (B,).
        """
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        # Use [CLS] representation.
        pooled = out.last_hidden_state[:, 0, :]  # (B, H)

        scores = self.head_ranking(pooled).squeeze(-1)  # (B,)
        esci_logits = self.head_esci(pooled)  # (B, 4)
        substitute_logits = self.head_substitute(pooled).squeeze(-1)  # (B,)

        return scores, esci_logits, substitute_logits

    def predict(
        self,
        pairs: List[List[str]],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> tuple[List[float], List[str], List[float]]:
        """
        Score (query, product) pairs and return ranking score, predicted ESCI
        class, and substitute probability per pair.

        Parameters
        ----------
        pairs : List[List[str]]
            List of [query, product_text] pairs.
        batch_size : int
            Batch size for forward passes.
        show_progress_bar : bool
            Whether to show a progress bar.

        Returns
        -------
        tuple[List[float], List[str], List[float]]
            (scores, esci_classes, substitute_probs), each of length len(pairs).
        """
        all_scores: List[float] = []
        all_esci: List[str] = []
        all_sub: List[float] = []

        if not pairs:
            return all_scores, all_esci, all_sub

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            scores, esci_labels, sub_probs = self._predict_batch(batch)
            all_scores.extend(scores)
            all_esci.extend(esci_labels)
            all_sub.extend(sub_probs)

        return all_scores, all_esci, all_sub

    def _predict_batch(
        self,
        batch_pairs: List[List[str]],
    ) -> tuple[List[float], List[str], List[float]]:
        """
        Predict scores, ESCI classes, and substitute probabilities for a batch.
        """
        enc = self.tokenizer(
            batch_pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        token_type_ids = enc.get("token_type_ids")

        self.eval()
        with torch.no_grad():
            scores, esci_logits, substitute_logits = self.forward(
                enc["input_ids"],
                enc["attention_mask"],
                token_type_ids,
            )

        pred_ids = esci_logits.argmax(dim=-1)
        sub_prob = torch.sigmoid(substitute_logits)

        score_list = [float(s) for s in scores.detach().cpu()]
        esci_list = [ESCI_ID2LABEL[int(i)] for i in pred_ids.detach().cpu()]
        sub_list = [float(p) for p in sub_prob.detach().cpu()]
        return score_list, esci_list, sub_list

    def rerank(
        self,
        query: str,
        candidates: List[tuple[str, str]],
        *,
        batch_size: int = 32,
    ) -> List[tuple[str, float, str, float]]:
        """
        Rerank (product_id, product_text) candidates for a single query.
        Returns (product_id, score, esci_class, substitute_prob) sorted by
        score descending.

        Parameters
        ----------
        query : str
            Query string.
        candidates : List[tuple[str, str]]
            List of (product_id, product_text) pairs.
        batch_size : int
            Batch size for scoring.

        Returns
        -------
        List[tuple[str, float, str, float]]
            List of (product_id, score, esci_class, substitute_prob) sorted by
            score descending.
        """
        if not candidates:
            return []
        pairs = [[query, text] for _pid, text in candidates]
        scores, esci_classes, sub_probs = self.predict(pairs, batch_size=batch_size)
        out = [(pid, float(sc), esc, float(sub)) for (pid, _), sc, esc, sub in zip(candidates, scores, esci_classes, sub_probs)]
        out.sort(key=lambda x: x[1], reverse=True)
        return out


def load_multi_task_reranker(
    model_path: str | Path | None = None,
    model_name: str = DEFAULT_RERANKER_MODEL,
    device: str | torch.device | None = None,
    cache_folder: str | Path | None = MODEL_CACHE_DIR,
) -> MultiTaskReranker:
    """
    Load multi-task learning reranker. If model_path exists, load from disk;
    else build from pretrained model_name.

    Parameters
    ----------
    model_path : str | Path | None
        Path to saved multi-task learning checkpoint directory. If it exists,
        load from here.
    model_name : str
        HuggingFace model id when model_path is not used.
    device : str | torch.device | None
        Device to load the model on.
    cache_folder : str | Path | None
        Cache directory for pretrained downloads.

    Returns
    -------
    MultiTaskReranker
        Loaded or newly constructed model.
    """
    path = Path(model_path) if model_path else None
    if path and path.exists():
        try:
            return MultiTaskReranker.from_pretrained(path, device=device)
        except Exception as e:
            logger.warning(
                "Could not load multi-task learning from %s (%s); using pretrained %s",
                path,
                e,
                model_name,
            )
    return MultiTaskReranker(model_name=model_name, device=device, cache_folder=cache_folder)
