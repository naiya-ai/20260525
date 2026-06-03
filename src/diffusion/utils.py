"""Utility functions shared by diffusion modules."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


LOG_EPS = 1e-30


def extract(values: Tensor, timesteps: Tensor, broadcast_shape: torch.Size) -> Tensor:
    """Extract timestep-specific values and reshape for broadcasting."""
    out = values.to(device=timesteps.device).gather(0, timesteps)
    while out.ndim < len(broadcast_shape):
        out = out.unsqueeze(-1)
    return out.expand(broadcast_shape)


def mean_flat(tensor: Tensor) -> Tensor:
    """Mean over all non-batch dimensions."""
    return tensor.mean(dim=tuple(range(1, tensor.ndim)))


def log_1_min_a(log_a: Tensor) -> Tensor:
    """Compute log(1 - exp(log_a)) stably for log_a <= 0."""
    return torch.log1p(-log_a.exp().clamp(max=1.0 - 1e-12))


def log_add_exp(a: Tensor, b: Tensor) -> Tensor:
    """Stable log(exp(a) + exp(b))."""
    maximum = torch.maximum(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def normal_kl(
    mean1: Tensor,
    logvar1: Tensor,
    mean2: Tensor | float,
    logvar2: Tensor | float,
) -> Tensor:
    """KL divergence between two diagonal Gaussian distributions."""
    if not torch.is_tensor(mean2):
        mean2 = torch.tensor(mean2, device=mean1.device, dtype=mean1.dtype)
    if not torch.is_tensor(logvar2):
        logvar2 = torch.tensor(logvar2, device=mean1.device, dtype=mean1.dtype)
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def index_to_log_onehot(indices: Tensor, category_sizes: np.ndarray) -> Tensor:
    """Convert categorical indices to concatenated log one-hot vectors."""
    if indices.ndim != 2:
        raise ValueError("indices must have shape (batch, n_cat).")
    if indices.shape[1] != len(category_sizes):
        raise ValueError(
            f"Expected {len(category_sizes)} categorical features, got {indices.shape[1]}."
        )

    parts = []
    for feature_index, size in enumerate(category_sizes.tolist()):
        values = indices[:, feature_index].long()
        if torch.any(values < 0) or torch.any(values >= size):
            raise ValueError(
                f"Categorical feature {feature_index} contains values outside 0..{size - 1}."
            )
        onehot = F.one_hot(values, num_classes=int(size)).to(dtype=torch.float32)
        parts.append(torch.log(onehot.clamp(min=LOG_EPS)))
    return torch.cat(parts, dim=1)


def log_onehot_to_index(log_x: Tensor, category_sizes: np.ndarray) -> Tensor:
    """Convert concatenated log one-hot/logit blocks to categorical indices."""
    indices = []
    start = 0
    for size in category_sizes.tolist():
        end = start + int(size)
        indices.append(log_x[:, start:end].argmax(dim=1, keepdim=True))
        start = end
    return torch.cat(indices, dim=1)


def sliced_logsumexp(values: Tensor, offsets: Tensor) -> Tensor:
    """Compute per-feature logsumexp and expand each result back to its slice."""
    outputs = torch.empty_like(values)
    for start, end in zip(offsets[:-1].tolist(), offsets[1:].tolist()):
        outputs[:, start:end] = values[:, start:end].logsumexp(
            dim=1,
            keepdim=True,
        )
    return outputs
