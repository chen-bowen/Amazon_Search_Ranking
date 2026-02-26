"""
Contrastive loss (InfoNCE) with in-batch negatives and optional hard-negative reweighting.

Harder negatives (higher query–product similarity) can be given slightly higher weight by
scaling their logits, but we keep a simple, stable formulation that works well in practice.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def contrastive_loss_with_reweighting(
    query_emb: torch.Tensor,
    product_emb: torch.Tensor,
    reweight_hard: bool = True,
    hard_weight_power: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    InfoNCE loss with in-batch negatives and a simple self-adversarial reweighting.

    Logits are raw dot products (L2-normalized ⇒ cosine similarity). Optional temperature scaling.

    Shapes
    ------
    query_emb:
        [B, D] – batch of B L2-normalized query embeddings.
    product_emb:
        [B, D] – batch of B L2-normalized product embeddings; row i is the positive for query i.

    Parameters
    ----------
    reweight_hard:
        If False, compute standard InfoNCE with no hard-negative reweighting.
    hard_weight_power:
        Strength of hard-negative reweighting. When > 0, off-diagonal logits are scaled
        by (1 + hard_weight_power) so that negatives contribute more.
    temperature:
        Softmax temperature. Effective logits are logits / temperature. Values < 1.0
        make the distribution sharper; values > 1.0 make it softer. Default: 1.0.

    Returns
    -------
    - loss : torch.Tensor
        Loss tensor.
    """
    B = query_emb.size(0)
    device = query_emb.device

    # Similarity matrix: [B, B] (cosine sim when embeddings are L2-normalized).
    logits = torch.mm(query_emb, product_emb.t())
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = logits / temperature

    # For each query i, the positive is product i (diagonal of logits).
    labels = torch.arange(B, device=device, dtype=torch.long)

    if not reweight_hard or hard_weight_power <= 0:
        # Standard InfoNCE: cross-entropy over all products in the batch.
        return F.cross_entropy(logits, labels)

    # Simple self-adversarial reweighting:
    # - scale only the off-diagonal (negative) logits by (1 + hard_weight_power),
    #   increasing the weight of negatives in the softmax denominator.
    mask_neg = ~torch.eye(B, dtype=torch.bool, device=device)
    logits_eff = logits.clone()
    logits_eff[mask_neg] = logits_eff[mask_neg] * (1.0 + hard_weight_power)

    return F.cross_entropy(logits_eff, labels)
