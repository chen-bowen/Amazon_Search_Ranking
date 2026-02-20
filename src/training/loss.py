"""
Contrastive loss (InfoNCE) with in-batch negatives and self-adversarial re-weighting.
Harder negatives (higher query-product similarity) get higher weight.
"""
from __future__ import annotations  # Enable postponed evaluation of type hints

import torch  # PyTorch tensor operations
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Functional API (e.g., cross_entropy)


def contrastive_loss_with_reweighting(
    query_emb: torch.Tensor,
    product_emb: torch.Tensor,
    temperature: float = 0.05,
    reweight_hard: bool = True,
    hard_weight_power: float = 1.0,
) -> torch.Tensor:
    """
    InfoNCE with in-batch negatives. For each query i, positive is product i; negatives are j != i.
    Self-adversarial: weight negative j by (exp(sim_ij/tau))^hard_weight_power so harder negatives count more.
    query_emb, product_emb: [B, D] L2-normalized.
    """
    B = query_emb.size(0)  # Batch size (number of query-product pairs)
    device = query_emb.device  # Device (CPU or GPU) of the embeddings
    # [B, B] similarity matrix (dot product = cosine for normalized)
    # Compute all pairwise similarities: query_emb @ product_emb^T
    logits = torch.mm(query_emb, product_emb.t()) / temperature  # Scale by temperature (lower = sharper distribution)
    # Labels: diagonal elements are positives (query i matches product i)
    labels = torch.arange(B, device=device, dtype=torch.long)  # [0, 1, 2, ..., B-1]
    if not reweight_hard or hard_weight_power <= 0:  # If reweighting disabled or power <= 0
        return F.cross_entropy(logits, labels)  # Standard InfoNCE loss (no reweighting)
    # Self-adversarial: weight off-diagonal (negative) logits by their probability
    # so harder negatives contribute more. New logits: diagonal unchanged, off-diagonal
    # scaled so that softmax denominator gets an extra factor for high-sim negatives.
    # We implement by weighting the negative exp terms: exp(logits_ij) -> exp(logits_ij)^(1+alpha)
    # i.e. logits_eff_ij = logits_ij * (1+alpha) for j != i. So we use effective logits:
    # for j != i: logits_eff[i,j] = logits[i,j] * (1 + hard_weight_power)
    # Then cross_entropy(logits_eff, labels) pushes hard negatives down more.
    # Create boolean mask: True for off-diagonal (negative pairs), False for diagonal (positive pairs)
    mask_neg = ~torch.eye(B, dtype=torch.bool, device=device)  # Invert identity matrix (True where i != j)
    logits_eff = logits.clone()  # Copy logits to avoid modifying original
    # Scale negative logits: higher similarity negatives get multiplied by (1 + hard_weight_power)
    # This makes the loss focus more on hard negatives (those the model currently scores highly)
    logits_eff[mask_neg] = logits_eff[mask_neg] * (1.0 + hard_weight_power)  # Apply scaling to negatives only
    return F.cross_entropy(logits_eff, labels)  # Compute cross-entropy loss with reweighted logits
