"""
Two-tower (bi-encoder) for query and product: same or sibling Sentence Transformer backbones, L2-normalized embeddings.
"""

from __future__ import annotations  # Enable postponed evaluation of type hints

from typing import List  # For type hints

import torch  # PyTorch tensor operations
import torch.nn as nn  # Neural network modules
from sentence_transformers import SentenceTransformer  # Pre-trained sentence encoder

# Name used by sentence_transformers.evaluation.InformationRetrievalEvaluator when score_functions is None.
SIMILARITY_FN_NAME = "cosine"


def _tokenize(encoder: SentenceTransformer, texts: List[str], device: torch.device) -> dict:
    """
    Tokenize texts using the encoder's tokenizer and move tensor fields to the target device.
    """
    features = encoder.tokenize(texts)
    # SentenceTransformer.tokenize returns a dict; keep only tensor values and move them to device.
    return {k: v.to(device) for k, v in features.items() if isinstance(v, torch.Tensor)}


class _MockModelCard:
    """Minimal stub so SentenceTransformers evaluators can attach metrics."""

    def set_evaluation_metrics(self, *args, **kwargs) -> None:  # type: ignore[override]
        return None


class TwoTowerEncoder(nn.Module):
    """
    Query and product towers with optional shared backbone.

    Notation used in comments:
    - N: number of text inputs passed to an encoder (e.g. number of queries at inference time).
    - B: batch size during training (number of (query, product) pairs in a mini‑batch).
    - D: embedding dimension of the SentenceTransformer backbone (e.g. 384 for MiniLM).

    All outputs are L2-normalized D‑dimensional vectors and we use dot product
    (equivalent to cosine for normalized vectors) as the similarity.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L12-v2",
        *,
        shared: bool = False,
        normalize: bool = True,
    ):
        super().__init__()
        self.normalize = normalize
        self.similarity_fn_name = SIMILARITY_FN_NAME
        # Some SentenceTransformers evaluators expect `model.model_card_data.set_evaluation_metrics(...)`.
        # Provide a cheap stub so they can call into it without errors.
        self.model_card_data = _MockModelCard()
        if shared:
            # Shared backbone: both towers use the same SentenceTransformer instance.
            self.query_encoder = SentenceTransformer(model_name)
            self.product_encoder = self.query_encoder
        else:
            # Non‑shared towers: independent encoders with the same architecture.
            self.query_encoder = SentenceTransformer(model_name)
            self.product_encoder = SentenceTransformer(model_name)

    @property
    def query_tokenizer(self):
        """Access tokenizer for query encoder."""
        return self.query_encoder.tokenizer  # Return the tokenizer from query encoder

    @property
    def product_tokenizer(self):
        """Access tokenizer for product encoder."""
        return self.product_encoder.tokenizer  # Return the tokenizer from product encoder

    def encode_queries(self, queries: List[str] | list, device: torch.device | None = None) -> torch.Tensor:
        """
        Encode query strings into embeddings (inference mode, no gradients).

        Shape:
        - Input: list of N query strings.
        - Output: tensor of shape [N, D] (one D‑dim embedding per query).
        """
        # Infer device either from model parameters if not provided
        if device is None:
            device = next(self.query_encoder.parameters()).device

        emb = self.query_encoder.encode(
            queries,
            device=str(device),
            convert_to_tensor=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return emb

    def encode_products(self, products: List[str] | list, device: torch.device | None = None) -> torch.Tensor:
        """
        Encode product strings into embeddings (inference mode, no gradients).

        Shape:
        - Input: list of N product strings.
        - Output: tensor of shape [N, D] (one D‑dim embedding per product).
        """
        # Infer device from model parameters if not provided
        if device is None:
            device = next(self.product_encoder.parameters()).device

        emb = self.product_encoder.encode(
            products,
            device=str(device),
            convert_to_tensor=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return emb

    def encode_query(
        self,
        sentences: List[str] | list,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_tensor: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Encode query strings (API expected by sentence_transformers.evaluation.InformationRetrievalEvaluator).
        Batches inputs and delegates to encode_queries.
        """
        device = next(self.query_encoder.parameters()).device
        out = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            emb = self.encode_queries(batch, device=device)
            out.append(emb)
        return torch.cat(out, dim=0)

    def encode_document(
        self,
        sentences: List[str] | list,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_tensor: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Encode document/product strings (API expected by InformationRetrievalEvaluator).
        Batches inputs and delegates to encode_products.
        """
        device = next(self.product_encoder.parameters()).device
        out = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            emb = self.encode_products(batch, device=device)
            out.append(emb)
        return torch.cat(out, dim=0)

    def forward(
        self,
        query_inputs: dict | None = None,
        product_inputs: dict | None = None,
        query_strings: List[str] | None = None,
        product_strings: List[str] | None = None,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass used during training.

        You can pass either:
        - tokenized inputs: `query_inputs` / `product_inputs` (dicts with tensor values),
        - or raw strings: `query_strings` / `product_strings`.

        Shapes:
        - B: batch size = number of (query, product) pairs.
        - q_emb: [B, D] query embeddings.
        - p_emb: [B, D] product embeddings.

        Both outputs are L2-normalized D‑dim vectors when `normalize=True`.
        """
        # Infer device either from tokenized inputs or from model parameters.
        if device is None and query_inputs is not None:
            device = next(query_inputs.values().__iter__()).device
        elif device is None:
            device = next(self.query_encoder.parameters()).device

        # If raw strings are provided, tokenize them and move tensors to the device.
        if query_strings is not None:
            query_inputs = _tokenize(self.query_encoder, query_strings, device)
        if product_strings is not None:
            product_inputs = _tokenize(self.product_encoder, product_strings, device)

        # Forward pass through each encoder to obtain sentence embeddings.
        q_emb = self.query_encoder(query_inputs)["sentence_embedding"]
        p_emb = self.product_encoder(product_inputs)["sentence_embedding"]

        if self.normalize:
            # L2-normalize so each embedding is a unit vector in ℝ^D.
            q_emb = nn.functional.normalize(q_emb, p=2, dim=-1)
            p_emb = nn.functional.normalize(p_emb, p=2, dim=-1)

        return q_emb, p_emb

    def similarity(self, query_emb: torch.Tensor, product_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between query and product embeddings using dot product
        (equivalent to cosine when embeddings are L2-normalized).

        Shapes:
        - query_emb: [Q, D] or [B, D].
        - product_emb:
            * [B, D]   -> one product per query (paired); returns [B].
            * [B, N, D] -> N candidates per query; returns [B, N].
            * [C, D]   with C != Q -> matrix of scores; returns [Q, C] (for InformationRetrievalEvaluator).

        Returns:
        - [B]     when product_emb is [B, D] and same batch size as query_emb.
        - [B, N]  when product_emb is [B, N, D].
        - [Q, C]  when query_emb is [Q, D] and product_emb is [C, D].
        """
        if product_emb.dim() == 3:
            return torch.einsum("bd,bnd->bn", query_emb, product_emb)
        if product_emb.dim() == 2 and query_emb.size(0) != product_emb.size(0):
            return query_emb @ product_emb.T
        return (query_emb * product_emb).sum(dim=-1)
