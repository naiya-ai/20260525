"""Denoising backbones for conditional tabular diffusion."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalTimeEmbedding(nn.Module):
    """Create sinusoidal timestep embeddings."""

    def __init__(self, embedding_dim: int, max_period: int = 10000) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if max_period <= 0:
            raise ValueError("max_period must be positive.")
        self.embedding_dim = int(embedding_dim)
        self.max_period = int(max_period)

    def forward(self, timesteps: Tensor) -> Tensor:
        if timesteps.ndim != 1:
            raise ValueError("timesteps must have shape (batch,).")
        half_dim = self.embedding_dim // 2
        frequencies = torch.exp(
            -math.log(self.max_period)
            * torch.arange(
                start=0,
                end=half_dim,
                dtype=torch.float32,
                device=timesteps.device,
            )
            / half_dim
        )
        args = timesteps[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        if self.embedding_dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])],
                dim=1,
            )
        return embedding


class MLPDenoiser(nn.Module):
    """Simple conditional MLP denoiser.

    The network does not project target or condition inputs separately. It
    concatenates noisy target values, source condition values, and sinusoidal
    timestep embeddings, then predicts a denoised target-sized output.
    """

    def __init__(
        self,
        *,
        target_dim: int,
        condition_dim: int,
        time_embedding_dim: int,
        hidden_layers: Sequence[int],
        dropout: float = 0.0,
        activation: str = "relu",
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        if target_dim <= 0:
            raise ValueError("target_dim must be positive.")
        if condition_dim < 0:
            raise ValueError("condition_dim must not be negative.")
        if time_embedding_dim <= 0:
            raise ValueError("time_embedding_dim must be positive.")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0, 1).")

        self.target_dim = int(target_dim)
        self.condition_dim = int(condition_dim)
        self.time_embedding_dim = int(time_embedding_dim)
        self.hidden_layers = [int(width) for width in hidden_layers]
        self.batch_norm = bool(batch_norm)
        if any(width <= 0 for width in self.hidden_layers):
            raise ValueError("All hidden layer widths must be positive.")

        self.time_embedding = SinusoidalTimeEmbedding(self.time_embedding_dim)
        input_dim = self.target_dim + self.condition_dim + self.time_embedding_dim

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for width in self.hidden_layers:
            layers.append(nn.Linear(previous_dim, width))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(_make_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_dim = width
        layers.append(nn.Linear(previous_dim, self.target_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        noisy_target: Tensor,
        timesteps: Tensor,
        *,
        condition: Tensor | None = None,
    ) -> Tensor:
        if noisy_target.ndim != 2:
            raise ValueError("noisy_target must have shape (batch, target_dim).")
        if noisy_target.shape[1] != self.target_dim:
            raise ValueError(
                f"Expected noisy_target dim {self.target_dim}, got {noisy_target.shape[1]}."
            )
        if timesteps.ndim != 1 or timesteps.shape[0] != noisy_target.shape[0]:
            raise ValueError("timesteps must have shape (batch,).")

        if self.condition_dim == 0:
            condition_values = noisy_target.new_empty((noisy_target.shape[0], 0))
        else:
            if condition is None:
                raise ValueError("condition is required when condition_dim > 0.")
            if condition.ndim != 2:
                raise ValueError("condition must have shape (batch, condition_dim).")
            if condition.shape != (noisy_target.shape[0], self.condition_dim):
                raise ValueError(
                    f"Expected condition shape "
                    f"{(noisy_target.shape[0], self.condition_dim)}, "
                    f"got {tuple(condition.shape)}."
                )
            condition_values = condition.to(dtype=noisy_target.dtype)

        time_values = self.time_embedding(timesteps).to(dtype=noisy_target.dtype)
        model_input = torch.cat(
            [noisy_target, condition_values, time_values],
            dim=1,
        )
        return self.net(model_input)


def _make_activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")
