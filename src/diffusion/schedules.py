"""Noise schedules for diffusion models."""

from __future__ import annotations

import math

import numpy as np


def get_named_beta_schedule(
    schedule_name: str,
    num_diffusion_timesteps: int,
) -> np.ndarray:
    """Return a beta schedule by name."""
    if num_diffusion_timesteps <= 0:
        raise ValueError("num_diffusion_timesteps must be positive.")

    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start,
            beta_end,
            num_diffusion_timesteps,
            dtype=np.float64,
        )

    if schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )

    raise ValueError(f"Unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(
    num_diffusion_timesteps: int,
    alpha_bar,
    max_beta: float = 0.999,
) -> np.ndarray:
    """Discretize an alpha_bar function into betas."""
    betas = []
    for index in range(num_diffusion_timesteps):
        t1 = index / num_diffusion_timesteps
        t2 = (index + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float64)
