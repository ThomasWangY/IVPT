"""Pixel-wise entropy loss to sharpen per-pixel prototype assignments."""

import torch


def pixel_wise_entropy_loss(maps):
    """
    Calculate pixel-wise entropy loss for a feature map
    :param maps: Attention map with shape (batch_size, channels, height, width) where channels is the landmark probability
    :return: value of the pixel-wise entropy loss
    """
    # Clamp to valid probability range for numerical stability (esp. under AMP / gumbel_softmax)
    maps = maps.float().clamp(min=1e-6, max=1.0)
    maps = maps / maps.sum(dim=1, keepdim=True)  # re-normalise after clamp
    # Calculate entropy for each pixel
    entropy = torch.distributions.categorical.Categorical(probs=maps.permute(0, 2, 3, 1).contiguous()).entropy()
    # Take the mean of the entropy
    return entropy.mean()
