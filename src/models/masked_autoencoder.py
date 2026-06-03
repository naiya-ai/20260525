"""Masked autoencoder for pretraining EDDI source encoders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.conditional_vae import _build_eddi_source_encoder, _make_mlp


class MaskedSourceAutoencoder(nn.Module):
    """Reconstruct source variables from an artificially masked source context."""

    def __init__(
        self,
        *,
        source_n_num_features: int,
        source_category_sizes: Sequence[int],
        source_encoder_config: Mapping[str, object],
        decoder_hidden_layers: Sequence[int] = (),
        dropout: float = 0.0,
        activation: str = "relu",
        batch_norm: bool = False,
        source_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if source_n_num_features < 0:
            raise ValueError("source_n_num_features must not be negative.")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0, 1).")
        self.source_n_num_features = int(source_n_num_features)
        self.source_category_sizes = [int(size) for size in source_category_sizes]
        if any(size <= 0 for size in self.source_category_sizes):
            raise ValueError("All source category sizes must be positive.")
        self.source_n_cat_features = len(self.source_category_sizes)
        self.source_cat_dim = sum(self.source_category_sizes)
        if self.source_n_num_features + self.source_n_cat_features <= 0:
            raise ValueError("At least one source feature is required.")

        self.source_encoder = (
            _build_eddi_source_encoder(
                n_num_features=source_n_num_features,
                category_sizes=source_category_sizes,
                config=source_encoder_config,
            )
            if source_encoder is None
            else source_encoder
        )
        if not hasattr(self.source_encoder, "output_dim"):
            raise ValueError("source_encoder must expose an output_dim attribute.")
        self.decoder = _make_mlp(
            input_dim=int(self.source_encoder.output_dim),
            hidden_layers=decoder_hidden_layers,
            output_dim=self.source_n_num_features + self.source_cat_dim,
            dropout=dropout,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.register_buffer(
            "source_category_sizes_tensor",
            torch.tensor(self.source_category_sizes, dtype=torch.long),
        )

    @property
    def condition_dim(self) -> int:
        return int(self.source_encoder.output_dim)

    def forward(
        self,
        *,
        source_num: Tensor | None,
        source_cat: Tensor | None,
        source_num_mask: Tensor | None,
        source_cat_mask: Tensor | None,
    ) -> dict[str, Tensor]:
        condition = self.source_encoder(
            source_num,
            source_cat,
            source_num_mask,
            source_cat_mask,
        )
        decoded = self.decoder(condition)
        num_mean = decoded[:, : self.source_n_num_features]
        cat_logits = decoded[:, self.source_n_num_features :]
        return {
            "condition": condition,
            "num_mean": num_mean,
            "cat_logits": cat_logits,
        }

    def loss(
        self,
        *,
        input_num: Tensor | None,
        input_cat: Tensor | None,
        input_num_mask: Tensor | None,
        input_cat_mask: Tensor | None,
        target_num: Tensor | None,
        target_cat: Tensor | None,
        loss_num_mask: Tensor | None,
        loss_cat_mask: Tensor | None,
    ) -> dict[str, Tensor]:
        outputs = self(
            source_num=input_num,
            source_cat=input_cat,
            source_num_mask=input_num_mask,
            source_cat_mask=input_cat_mask,
        )
        loss_num = self._num_reconstruction_loss(
            outputs["num_mean"],
            target_num,
            loss_num_mask,
        )
        loss_cat = self._cat_reconstruction_loss(
            outputs["cat_logits"],
            target_cat,
            loss_cat_mask,
        )
        observed_num_count = _feature_count(
            loss_num_mask,
            self.source_n_num_features,
            device=outputs["num_mean"].device,
            dtype=outputs["num_mean"].dtype,
        )
        observed_cat_count = _feature_count(
            loss_cat_mask,
            self.source_n_cat_features,
            device=outputs["num_mean"].device,
            dtype=outputs["num_mean"].dtype,
        )
        observed_feature_count = observed_num_count + observed_cat_count
        loss_reconstruction = loss_num + loss_cat
        return {
            "loss": loss_reconstruction,
            "loss_reconstruction": loss_reconstruction,
            "loss_num": loss_num,
            "loss_cat": loss_cat,
            "observed_feature_count": observed_feature_count,
            "observed_num_count": observed_num_count,
            "observed_cat_count": observed_cat_count,
            **outputs,
        }

    def _num_reconstruction_loss(
        self,
        num_mean: Tensor,
        target: Tensor | None,
        mask: Tensor | None,
    ) -> Tensor:
        if self.source_n_num_features == 0:
            return num_mean.new_zeros(())
        target = _matrix(target, "target_num", self.source_n_num_features).to(
            device=num_mean.device,
            dtype=num_mean.dtype,
        )
        mask = _matrix(mask, "loss_num_mask", self.source_n_num_features).to(
            device=num_mean.device,
            dtype=num_mean.dtype,
        ).clamp(0.0, 1.0)
        return _masked_mean((num_mean - target) ** 2, mask)

    def _cat_reconstruction_loss(
        self,
        cat_logits: Tensor,
        target: Tensor | None,
        mask: Tensor | None,
    ) -> Tensor:
        if self.source_n_cat_features == 0:
            return cat_logits.new_zeros(())
        target = _matrix(target, "target_cat", self.source_n_cat_features).to(
            device=cat_logits.device,
            dtype=torch.long,
        )
        mask = _matrix(mask, "loss_cat_mask", self.source_n_cat_features).to(
            device=cat_logits.device,
            dtype=cat_logits.dtype,
        ).clamp(0.0, 1.0)
        self._validate_source_cat_range(target, mask)

        per_feature = torch.empty(
            (cat_logits.shape[0], self.source_n_cat_features),
            device=cat_logits.device,
            dtype=cat_logits.dtype,
        )
        offset = 0
        for feature_index, size in enumerate(self.source_category_sizes):
            logits = cat_logits[:, offset : offset + size]
            safe_target = target[:, feature_index].clamp(min=0, max=size - 1)
            per_feature[:, feature_index] = F.cross_entropy(
                logits,
                safe_target,
                reduction="none",
            )
            offset += size
        return _masked_mean(per_feature, mask)

    def _validate_source_cat_range(self, values: Tensor, mask: Tensor) -> None:
        if torch.any(values < 0):
            raise ValueError("Categorical values must be non-negative class ids.")
        valid = mask.to(dtype=torch.bool)
        if not valid.any():
            return
        sizes = self.source_category_sizes_tensor.to(device=values.device).view(1, -1)
        invalid = valid & ((values == 0) | (values >= sizes))
        if torch.any(invalid):
            raise ValueError("Source categorical values contain invalid observed ids.")


def apply_source_feature_dropout(
    *,
    source_num: Tensor,
    source_cat: Tensor,
    source_num_mask: Tensor,
    source_cat_mask: Tensor,
    probability: float,
) -> dict[str, Tensor]:
    """Mask observed source features for denoising/reconstruction pretraining."""
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1].")
    num_observed = source_num_mask.clamp(0.0, 1.0)
    cat_observed = source_cat_mask.clamp(0.0, 1.0)
    if probability == 0:
        num_drop = torch.zeros_like(num_observed)
        cat_drop = torch.zeros_like(cat_observed)
    else:
        num_drop = (torch.rand_like(num_observed) < probability).to(num_observed.dtype)
        cat_drop = (torch.rand_like(cat_observed) < probability).to(cat_observed.dtype)
        num_drop = num_drop * num_observed
        cat_drop = cat_drop * cat_observed

    input_num_mask = num_observed * (1.0 - num_drop)
    input_cat_mask = cat_observed * (1.0 - cat_drop)
    input_num = source_num * input_num_mask
    input_cat = source_cat.masked_fill(input_cat_mask <= 0, 0)
    return {
        "input_num": input_num,
        "input_cat": input_cat,
        "input_num_mask": input_num_mask,
        "input_cat_mask": input_cat_mask,
        "loss_num_mask": num_drop,
        "loss_cat_mask": cat_drop,
    }


def _matrix(values: Tensor | None, name: str, expected_features: int) -> Tensor:
    if values is None:
        raise ValueError(f"{name} must not be None.")
    if values.ndim != 2 or values.shape[1] != expected_features:
        raise ValueError(
            f"{name} must have shape (batch, {expected_features}), got {tuple(values.shape)}."
        )
    return values


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weighted = values * mask
    return weighted.sum() / mask.sum().clamp_min(1.0)


def _feature_count(
    mask: Tensor | None,
    n_features: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if n_features == 0:
        return torch.zeros((), device=device, dtype=dtype)
    values = _matrix(mask, "mask", n_features).to(device=device, dtype=dtype).clamp(0.0, 1.0)
    return values.sum()
