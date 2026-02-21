"""Consistency loss (KL-divergence) between cross-layer prototype maps."""

import torch


def consistency_loss(pred, target, eps=1e-10):
    """Compute KL-divergence-based consistency loss.

    Encourages the prototype assignment maps at shallower layers to be
    consistent with the map at the deepest layer.

    Args:
        pred: Predicted map of shape ``(B, N+1, L)``.
        target: Target (detached) map of shape ``(B, N+1, L)``.
        eps: Small constant for numerical stability.

    Returns:
        Scalar consistency loss.
    """
    pred = pred + eps
    target = target + eps
    kl_loss = target * torch.log(target / pred)
    kl_loss = kl_loss.sum(dim=1)
    kl_loss = kl_loss.mean(dim=1)
    return kl_loss.mean()