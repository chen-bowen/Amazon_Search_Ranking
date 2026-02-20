"""
Two-tower (bi-encoder) for query and product: same or sibling Sentence Transformer backbones, L2-normalized embeddings.
"""
from __future__ import annotations  # Enable postponed evaluation of type hints

from typing import List  # For type hints

import torch  # PyTorch tensor operations
import torch.nn as nn  # Neural network modules
from sentence_transformers import SentenceTransformer  # Pre-trained sentence encoder


def _tokenize(encoder: SentenceTransformer, texts: List[str], device: torch.device) -> dict:
    """
    Tokenize texts using the encoder's tokenizer and move to device.
    """
    features = encoder.tokenize(texts)  # Tokenize texts into input_ids, attention_mask, etc.
    # SentenceTransformer.tokenize returns dict[str, Tensor]; move to device
    # Filter to only tensors (skip non-tensor metadata) and move to GPU/CPU
    return {k: v.to(device) for k, v in features.items() if isinstance(v, torch.Tensor)}


class TwoTowerEncoder(nn.Module):
    """
    Query and product towers with optional shared backbone.
    Outputs L2-normalized embeddings; similarity = dot product (or cosine).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        shared: bool = False,
        normalize: bool = True,
    ):
        super().__init__()  # Initialize parent nn.Module
        self.normalize = normalize  # Whether to L2-normalize embeddings
        if shared:  # If towers share parameters
            self.query_encoder = SentenceTransformer(model_name)  # Create encoder
            self.product_encoder = self.query_encoder  # Share same encoder for both towers
        else:  # If towers are separate
            self.query_encoder = SentenceTransformer(model_name)  # Create query encoder
            self.product_encoder = SentenceTransformer(model_name)  # Create separate product encoder

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
        """
        if device is None:  # If device not specified
            device = next(self.query_encoder.parameters()).device  # Use device from model parameters
        emb = self.query_encoder.encode(  # Encode queries to embeddings
            queries,  # List of query strings
            device=str(device),  # Device to run on ("cuda" or "cpu")
            convert_to_tensor=True,  # Return PyTorch tensor (not numpy)
            normalize_embeddings=self.normalize,  # L2-normalize if enabled
        )
        return emb  # Return [N, D] tensor of query embeddings

    def encode_products(self, products: List[str] | list, device: torch.device | None = None) -> torch.Tensor:
        """
        Encode product strings into embeddings (inference mode, no gradients).
        """
        if device is None:  # If device not specified
            device = next(self.product_encoder.parameters()).device  # Use device from model parameters
        emb = self.product_encoder.encode(  # Encode products to embeddings
            products,  # List of product text strings
            device=str(device),  # Device to run on ("cuda" or "cpu")
            convert_to_tensor=True,  # Return PyTorch tensor (not numpy)
            normalize_embeddings=self.normalize,  # L2-normalize if enabled
        )
        return emb  # Return [N, D] tensor of product embeddings

    def forward(
        self,
        query_inputs: dict | None = None,
        product_inputs: dict | None = None,
        query_strings: List[str] | None = None,
        product_strings: List[str] | None = None,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward: pass either tokenized (query_inputs, product_inputs) or raw (query_strings, product_strings).
        Returns (query_emb, product_emb) both [B, D] L2-normalized.
        Training mode: gradients enabled.
        """
        if device is None and query_inputs is not None:  # Infer device from tokenized inputs if available
            device = next(query_inputs.values().__iter__()).device  # Get device from first tensor value
        elif device is None:  # If still no device
            device = next(self.query_encoder.parameters()).device  # Use device from model parameters
        if query_strings is not None:  # If raw query strings provided
            query_inputs = _tokenize(self.query_encoder, query_strings, device)  # Tokenize and move to device
        if product_strings is not None:  # If raw product strings provided
            product_inputs = _tokenize(self.product_encoder, product_strings, device)  # Tokenize and move to device
        # Forward pass through query encoder: tokenized inputs -> embeddings
        q_emb = self.query_encoder(query_inputs)["sentence_embedding"]  # Extract sentence embedding from output
        # Forward pass through product encoder: tokenized inputs -> embeddings
        p_emb = self.product_encoder(product_inputs)["sentence_embedding"]  # Extract sentence embedding from output
        if self.normalize:  # If normalization enabled
            q_emb = nn.functional.normalize(q_emb, p=2, dim=-1)  # L2-normalize query embeddings (unit vectors)
            p_emb = nn.functional.normalize(p_emb, p=2, dim=-1)  # L2-normalize product embeddings (unit vectors)
        return q_emb, p_emb  # Return normalized embeddings [B, D] each

    def similarity(self, query_emb: torch.Tensor, product_emb: torch.Tensor) -> torch.Tensor:
        """
        Dot product (same as cosine when normalized). query_emb [B,D], product_emb [B,D] or [B,N,D].
        """
        if product_emb.dim() == 2:  # If product_emb is 2D [B, D] (one product per query)
            return (query_emb * product_emb).sum(dim=-1)  # Element-wise multiply and sum -> [B] similarities
        # If product_emb is 3D [B, N, D] (N products per query)
        return torch.einsum("bd,bnd->bn", query_emb, product_emb)  # Batch matrix multiplication -> [B, N] similarities
