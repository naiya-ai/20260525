"""Conditional VAE for mixed tabular targets with missing-value masks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.source_embedding import (
    make_eddi_source_encoder,
    make_mlp_source_encoder,
    make_tabular_feature_tokenizer,
)


class ConditionalMixedTypeVAE(nn.Module):
    """CVAE for numerical/categorical targets conditioned on source embeddings.

    The model is built for tabular data with many missing values:

    - source masks decide which source features contribute to the condition;
    - target masks are part of the posterior encoder context;
    - reconstruction losses are computed only over observed target features.
    """

    def __init__(
        self,
        *,
        source_n_num_features: int,
        source_category_sizes: Sequence[int],
        target_n_num_features: int,
        target_category_sizes: Sequence[int],
        source_encoder_config: Mapping[str, object],
        latent_dim: int,
        encoder_hidden_layers: Sequence[int] = (),
        prior_hidden_layers: Sequence[int] | None = None,
        decoder_hidden_layers: Sequence[int] = (),
        dropout: float = 0.0,
        activation: str = "relu",
        batch_norm: bool = False,
        normalization: str | None = None,
        source_encoder: nn.Module | None = None,
        prior_mode: str = "separate",
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if target_n_num_features < 0:
            raise ValueError("target_n_num_features must not be negative.")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0, 1).")
        prior_mode = prior_mode.lower()
        if prior_mode not in {"separate", "masked_target"}:
            raise ValueError("prior_mode must be either 'separate' or 'masked_target'.")
        self.prior_mode = prior_mode
        normalization = _resolve_mlp_normalization(
            batch_norm=batch_norm,
            normalization=normalization,
        )

        self.target_n_num_features = int(target_n_num_features)
        self.target_category_sizes = [int(size) for size in target_category_sizes]
        if any(size <= 0 for size in self.target_category_sizes):
            raise ValueError("All target category sizes must be positive.")
        self.target_n_cat_features = len(self.target_category_sizes)
        self.target_cat_dim = sum(self.target_category_sizes)
        if self.target_n_num_features + self.target_n_cat_features <= 0:
            raise ValueError("At least one target feature is required.")
        self.latent_dim = int(latent_dim)

        self.source_encoder = (
            _build_source_encoder(
                n_num_features=source_n_num_features,
                category_sizes=source_category_sizes,
                config=source_encoder_config,
            )
            if source_encoder is None
            else source_encoder
        )
        if not hasattr(self.source_encoder, "output_dim"):
            raise ValueError("source_encoder must expose an output_dim attribute.")
        condition_dim = int(self.source_encoder.output_dim)
        target_context_dim = (
            self.target_n_num_features
            + self.target_cat_dim
            + self.target_n_num_features
            + self.target_n_cat_features
        )
        self.target_context_dim = int(target_context_dim)

        self.encoder = _make_mlp(
            input_dim=condition_dim + target_context_dim,
            hidden_layers=encoder_hidden_layers,
            output_dim=2 * self.latent_dim,
            dropout=dropout,
            activation=activation,
            normalization=normalization,
        )
        if self.prior_mode == "separate":
            self.prior = _make_mlp(
                input_dim=condition_dim,
                hidden_layers=(
                    encoder_hidden_layers
                    if prior_hidden_layers is None
                    else prior_hidden_layers
                ),
                output_dim=2 * self.latent_dim,
                dropout=dropout,
                activation=activation,
                normalization=normalization,
            )
        else:
            self.prior = None
        self.decoder = _make_mlp(
            input_dim=condition_dim + self.latent_dim,
            hidden_layers=decoder_hidden_layers,
            output_dim=self.target_n_num_features + self.target_cat_dim,
            dropout=dropout,
            activation=activation,
            normalization=normalization,
        )
        self.register_buffer(
            "target_category_sizes_tensor",
            torch.tensor(self.target_category_sizes, dtype=torch.long),
        )

    @property
    def condition_dim(self) -> int:
        return int(self.source_encoder.output_dim)

    def encode_condition(
        self,
        *,
        source_num: Tensor | None,
        source_cat: Tensor | None,
        source_num_mask: Tensor | None,
        source_cat_mask: Tensor | None,
    ) -> Tensor:
        return self.source_encoder(
            source_num,
            source_cat,
            source_num_mask,
            source_cat_mask,
        )

    def encode_target_context(
        self,
        *,
        target_num: Tensor | None,
        target_cat: Tensor | None,
        target_num_mask: Tensor | None,
        target_cat_mask: Tensor | None,
    ) -> Tensor:
        batch_size = _infer_batch_size(
            target_num,
            target_cat,
            target_num_mask,
            target_cat_mask,
        )
        device = _infer_device(target_num, target_cat, target_num_mask, target_cat_mask)
        if self.target_n_num_features == 0:
            num_values = torch.empty((batch_size, 0), device=device, dtype=torch.float32)
            num_mask = torch.empty((batch_size, 0), device=device, dtype=torch.float32)
        else:
            num_values = _validate_matrix(
                target_num,
                "target_num",
                self.target_n_num_features,
            ).to(device=device, dtype=torch.float32)
            num_mask = _validate_matrix(
                target_num_mask,
                "target_num_mask",
                self.target_n_num_features,
            ).to(device=device, dtype=torch.float32).clamp(0.0, 1.0)

        if self.target_n_cat_features == 0:
            cat_values = torch.empty((batch_size, 0), device=device, dtype=torch.float32)
            cat_mask = torch.empty((batch_size, 0), device=device, dtype=torch.float32)
        else:
            target_cat = _validate_matrix(
                target_cat,
                "target_cat",
                self.target_n_cat_features,
            ).to(device=device, dtype=torch.long)
            cat_mask = _validate_matrix(
                target_cat_mask,
                "target_cat_mask",
                self.target_n_cat_features,
            ).to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
            self._validate_target_cat_range(target_cat, cat_mask)
            cat_values = self._target_cat_onehot(target_cat, cat_mask)

        return torch.cat([num_values, cat_values, num_mask, cat_mask], dim=1)

    def encode(
        self,
        *,
        source_num: Tensor | None,
        source_cat: Tensor | None,
        source_num_mask: Tensor | None,
        source_cat_mask: Tensor | None,
        target_num: Tensor | None,
        target_cat: Tensor | None,
        target_num_mask: Tensor | None,
        target_cat_mask: Tensor | None,
        context_target_num: Tensor | None = None,
        context_target_cat: Tensor | None = None,
        context_target_num_mask: Tensor | None = None,
        context_target_cat_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        condition = self.encode_condition(
            source_num=source_num,
            source_cat=source_cat,
            source_num_mask=source_num_mask,
            source_cat_mask=source_cat_mask,
        )
        target_context = self.encode_target_context(
            target_num=target_num if context_target_num is None else context_target_num,
            target_cat=target_cat if context_target_cat is None else context_target_cat,
            target_num_mask=(
                target_num_mask
                if context_target_num_mask is None
                else context_target_num_mask
            ),
            target_cat_mask=(
                target_cat_mask
                if context_target_cat_mask is None
                else context_target_cat_mask
            ),
        )
        parameters = self.encoder(torch.cat([target_context, condition], dim=1))
        mu, logvar = parameters.chunk(2, dim=1)
        return mu, logvar, condition

    def encode_prior(self, condition: Tensor) -> tuple[Tensor, Tensor]:
        if condition.ndim != 2 or condition.shape[1] != self.condition_dim:
            raise ValueError(
                f"condition must have shape (batch, {self.condition_dim}), "
                f"got {tuple(condition.shape)}."
            )
        if self.prior_mode == "separate":
            if self.prior is None:
                raise RuntimeError("Separate prior mode requires a prior network.")
            parameters = self.prior(condition)
        else:
            target_context = self._missing_target_context(condition)
            parameters = self.encoder(torch.cat([target_context, condition], dim=1))
        prior_mu, prior_logvar = parameters.chunk(2, dim=1)
        return prior_mu, prior_logvar

    def _missing_target_context(self, condition: Tensor) -> Tensor:
        return condition.new_zeros((condition.shape[0], self.target_context_dim))

    def reparameterize(
        self,
        mu: Tensor,
        logvar: Tensor,
        *,
        sample: bool | None = None,
    ) -> Tensor:
        if self.training if sample is None else sample:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z: Tensor, condition: Tensor) -> dict[str, Tensor]:
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"z must have shape (batch, {self.latent_dim}), got {tuple(z.shape)}."
            )
        if condition.ndim != 2 or condition.shape[0] != z.shape[0]:
            raise ValueError("condition must have shape (batch, condition_dim).")
        decoded = self.decoder(torch.cat([z, condition], dim=1))
        num_mean = decoded[:, : self.target_n_num_features]
        cat_logits = decoded[:, self.target_n_num_features :]
        return {"num_mean": num_mean, "cat_logits": cat_logits}

    def forward(
        self,
        *,
        source_num: Tensor | None,
        source_cat: Tensor | None,
        source_num_mask: Tensor | None,
        source_cat_mask: Tensor | None,
        target_num: Tensor | None,
        target_cat: Tensor | None,
        target_num_mask: Tensor | None,
        target_cat_mask: Tensor | None,
        context_target_num: Tensor | None = None,
        context_target_cat: Tensor | None = None,
        context_target_num_mask: Tensor | None = None,
        context_target_cat_mask: Tensor | None = None,
        sample_latent: bool | None = None,
    ) -> dict[str, Tensor]:
        mu, logvar, condition = self.encode(
            source_num=source_num,
            source_cat=source_cat,
            source_num_mask=source_num_mask,
            source_cat_mask=source_cat_mask,
            target_num=target_num,
            target_cat=target_cat,
            target_num_mask=target_num_mask,
            target_cat_mask=target_cat_mask,
            context_target_num=context_target_num,
            context_target_cat=context_target_cat,
            context_target_num_mask=context_target_num_mask,
            context_target_cat_mask=context_target_cat_mask,
        )
        z = self.reparameterize(mu, logvar, sample=sample_latent)
        prior_mu, prior_logvar = self.encode_prior(condition)
        decoded = self.decode(z, condition)
        return {
            "mu": mu,
            "logvar": logvar,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
            "z": z,
            "condition": condition,
            **decoded,
        }

    def sample_prior(self, condition: Tensor, *, sample: bool = True) -> Tensor:
        prior_mu, prior_logvar = self.encode_prior(condition)
        return self.reparameterize(prior_mu, prior_logvar, sample=sample)

    def loss(
        self,
        *,
        source_num: Tensor | None,
        source_cat: Tensor | None,
        source_num_mask: Tensor | None,
        source_cat_mask: Tensor | None,
        target_num: Tensor | None,
        target_cat: Tensor | None,
        target_num_mask: Tensor | None,
        target_cat_mask: Tensor | None,
        beta: float = 1.0,
        context_target_num: Tensor | None = None,
        context_target_cat: Tensor | None = None,
        context_target_num_mask: Tensor | None = None,
        context_target_cat_mask: Tensor | None = None,
        sample_latent: bool | None = None,
        sample_loss_weights: Tensor | None = None,
    ) -> dict[str, Tensor]:
        outputs = self(
            source_num=source_num,
            source_cat=source_cat,
            source_num_mask=source_num_mask,
            source_cat_mask=source_cat_mask,
            target_num=target_num,
            target_cat=target_cat,
            target_num_mask=target_num_mask,
            target_cat_mask=target_cat_mask,
            context_target_num=context_target_num,
            context_target_cat=context_target_cat,
            context_target_num_mask=context_target_num_mask,
            context_target_cat_mask=context_target_cat_mask,
            sample_latent=sample_latent,
        )
        sample_weights = _sample_weight_vector(
            sample_loss_weights,
            batch_size=outputs["mu"].shape[0],
            device=outputs["mu"].device,
            dtype=outputs["mu"].dtype,
        )
        loss_num = self._num_reconstruction_loss(
            outputs["num_mean"],
            target_num,
            target_num_mask,
            sample_weights,
        )
        loss_cat = self._cat_reconstruction_loss(
            outputs["cat_logits"],
            target_cat,
            target_cat_mask,
            sample_weights,
        )
        loss_reconstruction = loss_num + loss_cat
        loss_kl = _diagonal_gaussian_kl(
            outputs["mu"],
            outputs["logvar"],
            outputs["prior_mu"],
            outputs["prior_logvar"],
            sample_weights=sample_weights,
        )
        loss = loss_reconstruction + float(beta) * loss_kl
        observed_num_count = _masked_feature_count(
            target_num_mask,
            self.target_n_num_features,
            device=loss.device,
            dtype=loss.dtype,
            name="target_num_mask",
            sample_weights=sample_weights,
        )
        observed_cat_count = self._cat_observed_feature_count(
            target_cat,
            target_cat_mask,
            device=loss.device,
            dtype=loss.dtype,
            sample_weights=sample_weights,
        )
        observed_feature_count = observed_num_count + observed_cat_count
        batch_size = outputs["mu"].new_tensor(outputs["mu"].shape[0])
        loss_num_sum = loss_num * batch_size
        loss_cat_sum = loss_cat * batch_size
        loss_reconstruction_sum = loss_reconstruction * batch_size
        loss_kl_sum = loss_kl * batch_size
        loss_sum = loss * batch_size
        return {
            "loss": loss,
            "loss_reconstruction": loss_reconstruction,
            "loss_num": loss_num,
            "loss_cat": loss_cat,
            "loss_kl": loss_kl,
            "loss_per_observed_feature": _safe_divide(loss_sum, observed_feature_count),
            "loss_reconstruction_per_observed_feature": _safe_divide(
                loss_reconstruction_sum,
                observed_feature_count,
            ),
            "loss_num_per_observed_feature": _safe_divide(
                loss_num_sum,
                observed_num_count,
            ),
            "loss_cat_per_observed_feature": _safe_divide(
                loss_cat_sum,
                observed_cat_count,
            ),
            "loss_kl_per_observed_feature": _safe_divide(
                loss_kl_sum,
                observed_feature_count,
            ),
            "loss_observed_sum": loss_sum,
            "loss_reconstruction_observed_sum": loss_reconstruction_sum,
            "loss_num_observed_sum": loss_num_sum,
            "loss_cat_observed_sum": loss_cat_sum,
            "loss_kl_observed_sum": loss_kl_sum,
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
        sample_weights: Tensor | None = None,
    ) -> Tensor:
        if self.target_n_num_features == 0:
            return num_mean.new_zeros(())
        target = _validate_matrix(target, "target_num", self.target_n_num_features).to(
            device=num_mean.device,
            dtype=num_mean.dtype,
        )
        mask = _validate_matrix(mask, "target_num_mask", self.target_n_num_features).to(
            device=num_mean.device,
            dtype=num_mean.dtype,
        ).clamp(0.0, 1.0)
        return _masked_feature_sum_mean(
            (num_mean - target) ** 2,
            mask,
            sample_weights=sample_weights,
        )

    def _cat_reconstruction_loss(
        self,
        cat_logits: Tensor,
        target: Tensor | None,
        mask: Tensor | None,
        sample_weights: Tensor | None = None,
    ) -> Tensor:
        if self.target_n_cat_features == 0:
            return cat_logits.new_zeros(())
        target = _validate_matrix(target, "target_cat", self.target_n_cat_features).to(
            device=cat_logits.device,
            dtype=torch.long,
        )
        mask = _validate_matrix(mask, "target_cat_mask", self.target_n_cat_features).to(
            device=cat_logits.device,
            dtype=torch.float32,
        ).clamp(0.0, 1.0)
        self._validate_target_cat_range(target, mask)

        per_feature = torch.empty(
            (cat_logits.shape[0], self.target_n_cat_features),
            device=cat_logits.device,
            dtype=cat_logits.dtype,
        )
        offset = 0
        for feature_index, size in enumerate(self.target_category_sizes):
            logits = cat_logits[:, offset : offset + size]
            safe_target = target[:, feature_index].clamp(min=0, max=size - 1)
            per_feature[:, feature_index] = F.cross_entropy(
                logits,
                safe_target,
                reduction="none",
            )
            offset += size
        return _masked_feature_sum_mean(
            per_feature,
            mask,
            sample_weights=sample_weights,
        )

    def _cat_observed_feature_count(
        self,
        target: Tensor | None,
        mask: Tensor | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
        sample_weights: Tensor | None = None,
    ) -> Tensor:
        if self.target_n_cat_features == 0:
            return torch.zeros((), device=device, dtype=dtype)
        target = _validate_matrix(target, "target_cat", self.target_n_cat_features).to(
            device=device,
            dtype=torch.long,
        )
        mask_values = _validate_matrix(
            mask,
            "target_cat_mask",
            self.target_n_cat_features,
        ).to(device=device, dtype=dtype).clamp(0.0, 1.0)
        self._validate_target_cat_range(target, mask_values)

        observed_weights = mask_values
        if sample_weights is not None:
            sample_weights = _sample_weight_vector(
                sample_weights,
                batch_size=observed_weights.shape[0],
                device=device,
                dtype=dtype,
            )
            observed_weights = observed_weights * sample_weights.view(-1, 1)
        return observed_weights.sum()

    def _target_cat_onehot(self, target_cat: Tensor, cat_mask: Tensor) -> Tensor:
        encoded = []
        valid = cat_mask.to(dtype=torch.bool)
        for feature_index, size in enumerate(self.target_category_sizes):
            values = target_cat[:, feature_index].clamp(min=0, max=size - 1)
            onehot = F.one_hot(values, num_classes=size).to(dtype=torch.float32)
            onehot = onehot * valid[:, feature_index].unsqueeze(1).to(dtype=torch.float32)
            encoded.append(onehot)
        return torch.cat(encoded, dim=1)

    def _validate_target_cat_range(self, values: Tensor, mask: Tensor) -> None:
        if torch.any(values < 0):
            raise ValueError("Categorical values must be non-negative class ids.")
        valid = mask.to(dtype=torch.bool)
        if not valid.any():
            return
        sizes = self.target_category_sizes_tensor.to(device=values.device).view(1, -1)
        invalid = valid & ((values == 0) | (values >= sizes))
        if torch.any(invalid):
            raise ValueError("Target categorical values contain invalid observed ids.")


class RawMLPConditionalMixedTypeVAE(ConditionalMixedTypeVAE):
    """CVAE whose MLPs consume raw flattened source features directly.

    This backbone avoids a learned source encoder. Numerical source features are
    represented as ``value * valid_mask`` plus ``valid_mask``. Categorical source
    features are represented as mask-gated one-hot vectors plus ``valid_mask``.
    """

    def __init__(
        self,
        *,
        source_n_num_features: int,
        source_category_sizes: Sequence[int],
        target_n_num_features: int,
        target_category_sizes: Sequence[int],
        latent_dim: int,
        encoder_hidden_layers: Sequence[int] = (),
        prior_hidden_layers: Sequence[int] | None = None,
        decoder_hidden_layers: Sequence[int] = (),
        dropout: float = 0.0,
        activation: str = "relu",
        batch_norm: bool = False,
        normalization: str | None = None,
        prior_mode: str = "separate",
    ) -> None:
        nn.Module.__init__(self)
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if source_n_num_features < 0:
            raise ValueError("source_n_num_features must not be negative.")
        if target_n_num_features < 0:
            raise ValueError("target_n_num_features must not be negative.")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0, 1).")
        prior_mode = prior_mode.lower()
        if prior_mode not in {"separate", "masked_target"}:
            raise ValueError("prior_mode must be either 'separate' or 'masked_target'.")
        self.prior_mode = prior_mode
        normalization = _resolve_mlp_normalization(
            batch_norm=batch_norm,
            normalization=normalization,
        )

        self.source_n_num_features = int(source_n_num_features)
        self.source_category_sizes = [int(size) for size in source_category_sizes]
        if any(size <= 0 for size in self.source_category_sizes):
            raise ValueError("All source category sizes must be positive.")
        self.source_n_cat_features = len(self.source_category_sizes)
        if self.source_n_num_features + self.source_n_cat_features <= 0:
            raise ValueError("At least one source feature is required.")

        self.target_n_num_features = int(target_n_num_features)
        self.target_category_sizes = [int(size) for size in target_category_sizes]
        if any(size <= 0 for size in self.target_category_sizes):
            raise ValueError("All target category sizes must be positive.")
        self.target_n_cat_features = len(self.target_category_sizes)
        self.target_cat_dim = sum(self.target_category_sizes)
        if self.target_n_num_features + self.target_n_cat_features <= 0:
            raise ValueError("At least one target feature is required.")
        self.latent_dim = int(latent_dim)

        self._condition_dim = (
            2 * self.source_n_num_features
            + sum(self.source_category_sizes)
            + self.source_n_cat_features
        )
        target_context_dim = (
            self.target_n_num_features
            + self.target_cat_dim
            + self.target_n_num_features
            + self.target_n_cat_features
        )
        self.target_context_dim = int(target_context_dim)

        self.encoder = _make_mlp(
            input_dim=self._condition_dim + target_context_dim,
            hidden_layers=encoder_hidden_layers,
            output_dim=2 * self.latent_dim,
            dropout=dropout,
            activation=activation,
            normalization=normalization,
        )
        if self.prior_mode == "separate":
            self.prior = _make_mlp(
                input_dim=self._condition_dim,
                hidden_layers=(
                    encoder_hidden_layers
                    if prior_hidden_layers is None
                    else prior_hidden_layers
                ),
                output_dim=2 * self.latent_dim,
                dropout=dropout,
                activation=activation,
                normalization=normalization,
            )
        else:
            self.prior = None
        self.decoder = _make_mlp(
            input_dim=self._condition_dim + self.latent_dim,
            hidden_layers=decoder_hidden_layers,
            output_dim=self.target_n_num_features + self.target_cat_dim,
            dropout=dropout,
            activation=activation,
            normalization=normalization,
        )
        self.register_buffer(
            "source_category_sizes_tensor",
            torch.tensor(self.source_category_sizes, dtype=torch.long),
        )
        self.register_buffer(
            "target_category_sizes_tensor",
            torch.tensor(self.target_category_sizes, dtype=torch.long),
        )

    @property
    def condition_dim(self) -> int:
        return self._condition_dim

    def encode_condition(
        self,
        *,
        source_num: Tensor | None,
        source_cat: Tensor | None,
        source_num_mask: Tensor | None,
        source_cat_mask: Tensor | None,
    ) -> Tensor:
        return _raw_tabular_context(
            num_features=source_num,
            cat_features=source_cat,
            num_mask=source_num_mask,
            cat_mask=source_cat_mask,
            n_num_features=self.source_n_num_features,
            category_sizes=self.source_category_sizes,
            name="source",
        )


class TransformerConditionalMixedTypeVAE(ConditionalMixedTypeVAE):
    """CVAE with transformer latent token encoders and configurable decoder.

    The tokenizer is only the tabular embedding step. The trainable CVAE
    networks can be configured in two ways:

    - separate prior: source tokens + prior latent token -> prior;
    - masked-target prior: source tokens + zero target tokens + latent token -> prior;
    - transformer decoder: source tokens + latent code token + target query tokens
      -> target prediction.
    """

    def __init__(
        self,
        *,
        source_n_num_features: int,
        source_category_sizes: Sequence[int],
        target_n_num_features: int,
        target_category_sizes: Sequence[int],
        tokenizer_config: Mapping[str, object],
        transformer_config: Mapping[str, object],
        latent_dim: int,
        prior_config: Mapping[str, object] | None = None,
        decoder_config: Mapping[str, object] | None = None,
        decoder_hidden_layers: Sequence[int] = (),
        dropout: float = 0.0,
        activation: str = "relu",
        batch_norm: bool = False,
        normalization: str | None = None,
        prior_mode: str = "separate",
    ) -> None:
        nn.Module.__init__(self)
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if source_n_num_features < 0:
            raise ValueError("source_n_num_features must not be negative.")
        if target_n_num_features < 0:
            raise ValueError("target_n_num_features must not be negative.")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0, 1).")
        prior_mode = prior_mode.lower()
        if prior_mode not in {"separate", "masked_target"}:
            raise ValueError("prior_mode must be either 'separate' or 'masked_target'.")
        self.prior_mode = prior_mode
        normalization = _resolve_mlp_normalization(
            batch_norm=batch_norm,
            normalization=normalization,
        )
        decoder_config = {} if decoder_config is None else decoder_config
        decoder_type = str(decoder_config.get("type", "mlp")).lower()
        if decoder_type not in {"mlp", "transformer"}:
            raise ValueError("decoder.type must be either 'mlp' or 'transformer'.")
        self.decoder_type = decoder_type

        self.target_n_num_features = int(target_n_num_features)
        self.target_category_sizes = [int(size) for size in target_category_sizes]
        if any(size <= 0 for size in self.target_category_sizes):
            raise ValueError("All target category sizes must be positive.")
        self.target_n_cat_features = len(self.target_category_sizes)
        self.target_cat_dim = sum(self.target_category_sizes)
        if self.target_n_num_features + self.target_n_cat_features <= 0:
            raise ValueError("At least one target feature is required.")
        self.latent_dim = int(latent_dim)

        source_category_sizes = [int(size) for size in source_category_sizes]
        if any(size <= 0 for size in source_category_sizes):
            raise ValueError("All source category sizes must be positive.")
        tokenizer_type = str(tokenizer_config.get("type", "eddi")).lower()
        if tokenizer_type not in {"eddi", "raw_embedding", "raw_linear"}:
            raise ValueError(
                "model.tokenizer.type must be one of: 'eddi', 'raw_embedding', 'raw_linear'."
            )
        self.tokenizer_type = tokenizer_type
        token_dim = int(
            tokenizer_config.get(
                "out_dim",
                tokenizer_config.get(
                    "output_dim",
                    tokenizer_config.get("d_model", 128),
                ),
            )
        )
        if token_dim <= 0:
            raise ValueError("model.tokenizer.out_dim must be positive.")
        token_embedding_dim = (
            token_dim
            if tokenizer_type in {"raw_embedding", "raw_linear"}
            else int(tokenizer_config.get("embedding_dim", token_dim))
        )
        if token_embedding_dim <= 0:
            raise ValueError("model.tokenizer.embedding_dim must be positive.")
        self._condition_dim = token_dim
        self.token_embedding_dim = token_embedding_dim

        self.source_tokenizer, self.source_token_network = _make_cvae_tokenizer_stack(
            tokenizer_type=tokenizer_type,
            role="source",
            n_num_features=source_n_num_features,
            category_sizes=source_category_sizes,
            embedding_dim=token_embedding_dim,
            token_dim=token_dim,
        )
        self.target_tokenizer, self.target_token_network = _make_cvae_tokenizer_stack(
            tokenizer_type=tokenizer_type,
            role="target",
            n_num_features=self.target_n_num_features,
            category_sizes=self.target_category_sizes,
            embedding_dim=token_embedding_dim,
            token_dim=token_dim,
        )
        self.encoder = _TransformerLatentEncoder(
            token_dim=token_dim,
            config=transformer_config,
            latent_dim=self.latent_dim,
            default_dropout=dropout,
        )
        if self.prior_mode == "separate":
            self.prior = _TransformerLatentPrior(
                token_dim=token_dim,
                config=transformer_config if prior_config is None else prior_config,
                latent_dim=self.latent_dim,
                default_dropout=dropout,
            )
        else:
            self.prior = None
        if self.decoder_type == "transformer":
            (
                self.decoder_source_tokenizer,
                self.decoder_source_token_network,
            ) = _make_cvae_tokenizer_stack(
                tokenizer_type=tokenizer_type,
                role="decoder_source",
                n_num_features=source_n_num_features,
                category_sizes=source_category_sizes,
                embedding_dim=token_embedding_dim,
                token_dim=token_dim,
            )
            self.decoder = _TransformerTabularDecoder(
                token_dim=token_dim,
                latent_dim=self.latent_dim,
                target_n_num_features=self.target_n_num_features,
                target_category_sizes=self.target_category_sizes,
                config=decoder_config,
                default_dropout=dropout,
            )
        else:
            self.decoder_source_tokenizer = None
            self.decoder_source_token_network = None
            self.decoder = _make_mlp(
                input_dim=token_dim + self.latent_dim,
                hidden_layers=decoder_hidden_layers,
                output_dim=self.target_n_num_features + self.target_cat_dim,
                dropout=dropout,
                activation=activation,
                normalization=normalization,
            )
        self.register_buffer(
            "target_category_sizes_tensor",
            torch.tensor(self.target_category_sizes, dtype=torch.long),
        )

    @property
    def condition_dim(self) -> int:
        return self._condition_dim

    def encode_condition(
        self,
        *,
        source_num: Tensor | None,
        source_cat: Tensor | None,
        source_num_mask: Tensor | None,
        source_cat_mask: Tensor | None,
    ) -> Tensor:
        source_tokens, _ = self._source_tokens(
            source_num=source_num,
            source_cat=source_cat,
            source_num_mask=source_num_mask,
            source_cat_mask=source_cat_mask,
        )
        _, _, condition = self._prior_from_source_tokens(source_tokens)
        return condition

    def encode(
        self,
        *,
        source_num: Tensor | None,
        source_cat: Tensor | None,
        source_num_mask: Tensor | None,
        source_cat_mask: Tensor | None,
        target_num: Tensor | None,
        target_cat: Tensor | None,
        target_num_mask: Tensor | None,
        target_cat_mask: Tensor | None,
        context_target_num: Tensor | None = None,
        context_target_cat: Tensor | None = None,
        context_target_num_mask: Tensor | None = None,
        context_target_cat_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        source_tokens, _ = self._source_tokens(
            source_num=source_num,
            source_cat=source_cat,
            source_num_mask=source_num_mask,
            source_cat_mask=source_cat_mask,
        )
        target_tokens, _ = self._target_tokens(
            num_features=target_num if context_target_num is None else context_target_num,
            cat_features=target_cat if context_target_cat is None else context_target_cat,
            num_mask=(
                target_num_mask
                if context_target_num_mask is None
                else context_target_num_mask
            ),
            cat_mask=(
                target_cat_mask
                if context_target_cat_mask is None
                else context_target_cat_mask
            ),
        )
        mu, logvar, _ = self.encoder.forward_with_context(source_tokens, target_tokens)
        _, _, condition = self._prior_from_source_tokens(source_tokens)
        return mu, logvar, condition

    def forward(
        self,
        *,
        source_num: Tensor | None,
        source_cat: Tensor | None,
        source_num_mask: Tensor | None,
        source_cat_mask: Tensor | None,
        target_num: Tensor | None,
        target_cat: Tensor | None,
        target_num_mask: Tensor | None,
        target_cat_mask: Tensor | None,
        context_target_num: Tensor | None = None,
        context_target_cat: Tensor | None = None,
        context_target_num_mask: Tensor | None = None,
        context_target_cat_mask: Tensor | None = None,
        sample_latent: bool | None = None,
    ) -> dict[str, Tensor]:
        source_tokens, _ = self._source_tokens(
            source_num=source_num,
            source_cat=source_cat,
            source_num_mask=source_num_mask,
            source_cat_mask=source_cat_mask,
        )
        target_tokens, _ = self._target_tokens(
            num_features=target_num if context_target_num is None else context_target_num,
            cat_features=target_cat if context_target_cat is None else context_target_cat,
            num_mask=(
                target_num_mask
                if context_target_num_mask is None
                else context_target_num_mask
            ),
            cat_mask=(
                target_cat_mask
                if context_target_cat_mask is None
                else context_target_cat_mask
            ),
        )
        mu, logvar, _ = self.encoder.forward_with_context(source_tokens, target_tokens)
        prior_mu, prior_logvar, condition = self._prior_from_source_tokens(source_tokens)
        z = self.reparameterize(mu, logvar, sample=sample_latent)
        decoded = self.decode_from_source(
            z,
            source_num=source_num,
            source_cat=source_cat,
            source_num_mask=source_num_mask,
            source_cat_mask=source_cat_mask,
            condition=condition,
        )
        return {
            "mu": mu,
            "logvar": logvar,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
            "z": z,
            "condition": condition,
            **decoded,
        }

    def encode_prior(self, condition: Tensor) -> tuple[Tensor, Tensor]:
        if condition.ndim != 2 or condition.shape[1] != self.condition_dim:
            raise ValueError(
                f"condition must have shape (batch, {self.condition_dim}), "
                f"got {tuple(condition.shape)}."
            )
        if self.prior_mode == "separate":
            if self.prior is None:
                raise RuntimeError("Separate prior mode requires a prior network.")
            parameters = self.prior.project_context(condition)
        else:
            parameters = self.encoder.project_context(condition)
        prior_mu, prior_logvar = parameters.chunk(2, dim=1)
        return prior_mu, prior_logvar

    def decode(self, z: Tensor, condition: Tensor) -> dict[str, Tensor]:
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"z must have shape (batch, {self.latent_dim}), got {tuple(z.shape)}."
            )
        if condition.ndim != 2 or condition.shape[0] != z.shape[0]:
            raise ValueError("condition must have shape (batch, condition_dim).")
        if self.decoder_type == "mlp":
            decoded = self.decoder(torch.cat([z, condition], dim=1))
            num_mean = decoded[:, : self.target_n_num_features]
            cat_logits = decoded[:, self.target_n_num_features :]
            return {"num_mean": num_mean, "cat_logits": cat_logits}
        return self.decoder(z=z, source_tokens=condition.unsqueeze(1))

    def decode_from_source(
        self,
        z: Tensor,
        *,
        source_num: Tensor | None,
        source_cat: Tensor | None,
        source_num_mask: Tensor | None,
        source_cat_mask: Tensor | None,
        condition: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if self.decoder_type == "mlp":
            if condition is None:
                condition = self.encode_condition(
                    source_num=source_num,
                    source_cat=source_cat,
                    source_num_mask=source_num_mask,
                    source_cat_mask=source_cat_mask,
                )
            return self.decode(z, condition)
        decoder_source_tokens = self._decoder_source_tokens(
            source_num=source_num,
            source_cat=source_cat,
            source_num_mask=source_num_mask,
            source_cat_mask=source_cat_mask,
        )
        return self.decoder(z=z, source_tokens=decoder_source_tokens)

    def _prior_from_source_tokens(self, source_tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.prior_mode == "separate":
            if self.prior is None:
                raise RuntimeError("Separate prior mode requires a prior network.")
            return self.prior(source_tokens)
        target_tokens = self._missing_target_tokens(
            batch_size=source_tokens.shape[0],
            device=source_tokens.device,
            dtype=source_tokens.dtype,
        )
        return self.encoder.forward_with_context(source_tokens, target_tokens)

    def _missing_target_tokens(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        target_num = None
        target_num_mask = None
        if self.target_n_num_features > 0:
            target_num = torch.zeros(
                (batch_size, self.target_n_num_features),
                device=device,
                dtype=dtype,
            )
            target_num_mask = torch.zeros_like(target_num)

        target_cat = None
        target_cat_mask = None
        if self.target_n_cat_features > 0:
            target_cat = torch.zeros(
                (batch_size, self.target_n_cat_features),
                device=device,
                dtype=torch.long,
            )
            target_cat_mask = torch.zeros(
                (batch_size, self.target_n_cat_features),
                device=device,
                dtype=dtype,
            )

        target_tokens, _ = self._target_tokens(
            num_features=target_num,
            cat_features=target_cat,
            num_mask=target_num_mask,
            cat_mask=target_cat_mask,
        )
        return target_tokens

    def _source_tokens(
        self,
        *,
        source_num: Tensor | None,
        source_cat: Tensor | None,
        source_num_mask: Tensor | None,
        source_cat_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        tokens, masks = self.source_tokenizer(
            num_features=source_num,
            cat_features=source_cat,
            num_mask=source_num_mask,
            cat_mask=source_cat_mask,
        )
        return self._project_tokens(tokens, masks, self.source_token_network)

    def _target_tokens(
        self,
        *,
        num_features: Tensor | None,
        cat_features: Tensor | None,
        num_mask: Tensor | None,
        cat_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        tokens, masks = self.target_tokenizer(
            num_features=num_features,
            cat_features=cat_features,
            num_mask=num_mask,
            cat_mask=cat_mask,
        )
        return self._project_tokens(tokens, masks, self.target_token_network)

    def _decoder_source_tokens(
        self,
        *,
        source_num: Tensor | None,
        source_cat: Tensor | None,
        source_num_mask: Tensor | None,
        source_cat_mask: Tensor | None,
    ) -> Tensor:
        if self.decoder_source_tokenizer is None:
            raise RuntimeError("decoder_source_tokenizer is only available for transformer decoder.")
        tokens, masks = self.decoder_source_tokenizer(
            num_features=source_num,
            cat_features=source_cat,
            num_mask=source_num_mask,
            cat_mask=source_cat_mask,
        )
        if self.decoder_source_token_network is None:
            raise RuntimeError("decoder_source_token_network is only available for transformer decoder.")
        projected, _ = self._project_tokens(
            tokens,
            masks,
            self.decoder_source_token_network,
        )
        return projected

    @staticmethod
    def _project_tokens(
        tokens: Tensor,
        masks: Tensor,
        token_network: nn.Module,
    ) -> tuple[Tensor, Tensor]:
        projected = token_network(tokens)
        projected = projected * masks.unsqueeze(-1)
        return projected, masks


class _TransformerLatentEncoder(nn.Module):
    """Latent-token encoder for posterior and masked-target prior."""

    def __init__(
        self,
        *,
        token_dim: int,
        config: Mapping[str, object],
        latent_dim: int,
        default_dropout: float,
    ) -> None:
        super().__init__()
        self.latent_token = nn.Parameter(torch.empty(1, 1, token_dim))
        self.transformer = _make_transformer_encoder(
            token_dim=token_dim,
            config=config,
            default_dropout=default_dropout,
        )
        self.projection = nn.Linear(token_dim, 2 * latent_dim)
        nn.init.normal_(self.latent_token, std=0.02)

    def forward(self, source_tokens: Tensor, target_tokens: Tensor) -> tuple[Tensor, Tensor]:
        mu, logvar, _ = self.forward_with_context(source_tokens, target_tokens)
        return mu, logvar

    def forward_with_context(
        self,
        source_tokens: Tensor,
        target_tokens: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = source_tokens.shape[0]
        latent_token = self.latent_token.expand(batch_size, -1, -1)
        encoded = self.transformer(
            torch.cat([source_tokens, target_tokens, latent_token], dim=1)
        )
        context = encoded[:, -1, :]
        parameters = self.project_context(context)
        mu, logvar = parameters.chunk(2, dim=1)
        return mu, logvar, context

    def project_context(self, context: Tensor) -> Tensor:
        return self.projection(context)


class _TransformerLatentPrior(nn.Module):
    """Prior network p(z | source tokens)."""

    def __init__(
        self,
        *,
        token_dim: int,
        config: Mapping[str, object],
        latent_dim: int,
        default_dropout: float,
    ) -> None:
        super().__init__()
        self.latent_token = nn.Parameter(torch.empty(1, 1, token_dim))
        self.transformer = _make_transformer_encoder(
            token_dim=token_dim,
            config=config,
            default_dropout=default_dropout,
        )
        self.projection = nn.Linear(token_dim, 2 * latent_dim)
        nn.init.normal_(self.latent_token, std=0.02)

    def forward(self, source_tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = source_tokens.shape[0]
        latent_token = self.latent_token.expand(batch_size, -1, -1)
        encoded = self.transformer(torch.cat([source_tokens, latent_token], dim=1))
        context = encoded[:, -1, :]
        parameters = self.project_context(context)
        prior_mu, prior_logvar = parameters.chunk(2, dim=1)
        return prior_mu, prior_logvar, context

    def project_context(self, context: Tensor) -> Tensor:
        return self.projection(context)


class _TransformerTabularDecoder(nn.Module):
    """Decode target feature query tokens from source tokens and a latent code."""

    def __init__(
        self,
        *,
        token_dim: int,
        latent_dim: int,
        target_n_num_features: int,
        target_category_sizes: Sequence[int],
        config: Mapping[str, object],
        default_dropout: float,
    ) -> None:
        super().__init__()
        self.target_n_num_features = int(target_n_num_features)
        self.target_category_sizes = [int(size) for size in target_category_sizes]
        if self.target_n_num_features < 0:
            raise ValueError("target_n_num_features must not be negative.")
        if any(size <= 0 for size in self.target_category_sizes):
            raise ValueError("All target category sizes must be positive.")
        self.target_n_cat_features = len(self.target_category_sizes)
        self.target_n_tokens = self.target_n_num_features + self.target_n_cat_features
        if self.target_n_tokens <= 0:
            raise ValueError("At least one target feature is required.")

        self.latent_to_token = nn.Linear(latent_dim, token_dim)
        self.latent_role_embedding = nn.Parameter(torch.empty(1, 1, token_dim))
        self.latent_norm = nn.LayerNorm(token_dim)
        self.target_query_embedding = nn.Parameter(
            torch.empty(1, self.target_n_tokens, token_dim)
        )
        self.transformer = _make_transformer_encoder(
            token_dim=token_dim,
            config=config,
            default_dropout=default_dropout,
        )
        self.num_head = (
            nn.Linear(token_dim, 1)
            if self.target_n_num_features > 0
            else None
        )
        self.cat_heads = nn.ModuleList(
            nn.Linear(token_dim, size)
            for size in self.target_category_sizes
        )
        nn.init.normal_(self.latent_role_embedding, std=0.02)
        nn.init.normal_(self.target_query_embedding, std=0.02)

    def forward(self, *, z: Tensor, source_tokens: Tensor) -> dict[str, Tensor]:
        if z.ndim != 2:
            raise ValueError("z must have shape (batch, latent_dim).")
        if source_tokens.ndim != 3:
            raise ValueError("source_tokens must have shape (batch, tokens, token_dim).")
        batch_size = source_tokens.shape[0]
        if z.shape[0] != batch_size:
            raise ValueError("z and source_tokens must have the same batch size.")

        latent_token = self.latent_to_token(z).unsqueeze(1)
        latent_token = self.latent_norm(latent_token + self.latent_role_embedding)
        target_queries = self.target_query_embedding.expand(batch_size, -1, -1)
        encoded = self.transformer(
            torch.cat([source_tokens, latent_token, target_queries], dim=1)
        )
        target_outputs = encoded[:, -self.target_n_tokens :, :]

        if self.target_n_num_features > 0:
            if self.num_head is None:
                raise RuntimeError("num_head is not configured.")
            num_mean = self.num_head(
                target_outputs[:, : self.target_n_num_features, :]
            ).squeeze(-1)
        else:
            num_mean = target_outputs.new_empty((batch_size, 0))

        cat_logits = []
        offset = self.target_n_num_features
        for feature_index, head in enumerate(self.cat_heads):
            feature_token = target_outputs[:, offset + feature_index, :]
            cat_logits.append(head(feature_token))
        if cat_logits:
            cat_logits_tensor = torch.cat(cat_logits, dim=1)
        else:
            cat_logits_tensor = target_outputs.new_empty((batch_size, 0))
        return {"num_mean": num_mean, "cat_logits": cat_logits_tensor}


def build_conditional_vae_model(
    *,
    source_n_num_features: int,
    source_category_sizes: Sequence[int],
    target_n_num_features: int,
    target_category_sizes: Sequence[int],
    model_config: Mapping[str, object],
) -> ConditionalMixedTypeVAE:
    """Build the configured CVAE backbone."""

    backbone = str(model_config.get("backbone", _infer_backbone(model_config))).lower()
    if backbone == "eddi":
        return ConditionalMixedTypeVAE(
            source_n_num_features=source_n_num_features,
            source_category_sizes=source_category_sizes,
            target_n_num_features=target_n_num_features,
            target_category_sizes=target_category_sizes,
            source_encoder_config=_mapping_config(model_config, "source_encoder"),
            latent_dim=int(model_config["latent_dim"]),
            encoder_hidden_layers=_sequence_config(
                model_config.get("encoder_hidden_layers", [])
            ),
            prior_hidden_layers=_optional_sequence_config(
                model_config.get("prior_hidden_layers")
            ),
            decoder_hidden_layers=_sequence_config(
                model_config.get("decoder_hidden_layers", [])
            ),
            dropout=float(model_config.get("dropout", 0.0)),
            activation=str(model_config.get("activation", "relu")),
            batch_norm=bool(model_config.get("batch_norm", False)),
            normalization=_optional_normalization_config(model_config),
            prior_mode=str(model_config.get("prior_mode", "separate")),
        )
    if backbone == "raw_mlp":
        return RawMLPConditionalMixedTypeVAE(
            source_n_num_features=source_n_num_features,
            source_category_sizes=source_category_sizes,
            target_n_num_features=target_n_num_features,
            target_category_sizes=target_category_sizes,
            latent_dim=int(model_config["latent_dim"]),
            encoder_hidden_layers=_sequence_config(
                model_config.get("encoder_hidden_layers", [])
            ),
            prior_hidden_layers=_optional_sequence_config(
                model_config.get("prior_hidden_layers")
            ),
            decoder_hidden_layers=_sequence_config(
                model_config.get("decoder_hidden_layers", [])
            ),
            dropout=float(model_config.get("dropout", 0.0)),
            activation=str(model_config.get("activation", "relu")),
            batch_norm=bool(model_config.get("batch_norm", False)),
            normalization=_optional_normalization_config(model_config),
            prior_mode=str(model_config.get("prior_mode", "separate")),
        )
    if backbone == "transformer":
        tokenizer_config = _transformer_tokenizer_config(model_config)
        encoder_config = _transformer_encoder_config(model_config)
        prior_config = _transformer_prior_config(model_config)
        return TransformerConditionalMixedTypeVAE(
            source_n_num_features=source_n_num_features,
            source_category_sizes=source_category_sizes,
            target_n_num_features=target_n_num_features,
            target_category_sizes=target_category_sizes,
            tokenizer_config=tokenizer_config,
            transformer_config=encoder_config,
            latent_dim=int(model_config["latent_dim"]),
            prior_config=prior_config,
            decoder_config=_mapping_config(model_config, "decoder")
            if "decoder" in model_config
            else None,
            decoder_hidden_layers=_sequence_config(
                model_config.get("decoder_hidden_layers", [])
            ),
            dropout=float(model_config.get("dropout", 0.0)),
            activation=str(model_config.get("activation", "relu")),
            batch_norm=bool(model_config.get("batch_norm", False)),
            normalization=_optional_normalization_config(model_config),
            prior_mode=str(model_config.get("prior_mode", "separate")),
        )
    raise ValueError(f"Unsupported model.backbone: {backbone!r}")


def _infer_backbone(model_config: Mapping[str, object]) -> str:
    encoder_config = _mapping_config(model_config, "encoder") if "encoder" in model_config else {}
    if str(encoder_config.get("type", "")).lower() == "transformer":
        return "transformer"
    if "transformer" in model_config:
        return "transformer"
    return "eddi"


def _transformer_tokenizer_config(model_config: Mapping[str, object]) -> dict[str, object]:
    tokenizer_config = dict(_mapping_config(model_config, "tokenizer"))
    tokenizer_type = str(tokenizer_config.get("type", "eddi")).lower()
    if tokenizer_type not in {"eddi", "raw_embedding", "raw_linear"}:
        raise ValueError(
            "model.tokenizer.type must be one of: 'eddi', 'raw_embedding', 'raw_linear'."
        )
    if tokenizer_config.get("layers") not in (None, []):
        raise ValueError("model.tokenizer.layers is not supported; use embedding_dim and out_dim.")
    if tokenizer_config.get("hidden_layers") not in (None, []):
        raise ValueError("model.tokenizer.hidden_layers is not supported; use embedding_dim and out_dim.")
    transformer_config = _transformer_encoder_config(model_config)
    prior_config = _transformer_prior_config(model_config)
    decoder_config = _mapping_config(model_config, "decoder") if "decoder" in model_config else {}
    configured_output_dims = []
    if "d_model" in model_config:
        configured_output_dims.append(("model.d_model", int(model_config["d_model"])))
    if "out_dim" in tokenizer_config:
        configured_output_dims.append(
            ("model.tokenizer.out_dim", int(tokenizer_config["out_dim"]))
        )
    if "output_dim" in tokenizer_config:
        configured_output_dims.append(
            ("model.tokenizer.output_dim", int(tokenizer_config["output_dim"]))
        )
    if "d_model" in tokenizer_config:
        configured_output_dims.append(
            ("model.tokenizer.d_model", int(tokenizer_config["d_model"]))
        )
    if "d_model" in transformer_config:
        configured_output_dims.append(
            ("model.encoder.d_model", int(transformer_config["d_model"]))
        )
    if prior_config is not None and "d_model" in prior_config:
        configured_output_dims.append(
            ("model.prior.d_model", int(prior_config["d_model"]))
        )
    if "d_model" in decoder_config:
        configured_output_dims.append(
            ("model.decoder.d_model", int(decoder_config["d_model"]))
        )

    embedding_dim_was_configured = "embedding_dim" in tokenizer_config
    if configured_output_dims:
        first_name, first_value = configured_output_dims[0]
        for name, value in configured_output_dims[1:]:
            if value != first_value:
                raise ValueError(
                    f"Transformer token dimension mismatch: {first_name}={first_value}, "
                    f"{name}={value}."
                )
        tokenizer_config["out_dim"] = first_value
        if not embedding_dim_was_configured:
            tokenizer_config["embedding_dim"] = first_value
    else:
        embedding_dim = int(tokenizer_config.get("embedding_dim", 128))
        tokenizer_config["embedding_dim"] = embedding_dim
        tokenizer_config["out_dim"] = int(
            tokenizer_config.get(
                "out_dim",
                tokenizer_config.get("output_dim", embedding_dim),
            )
        )
    return tokenizer_config


class _RawEmbeddingFeatureTokenizer(nn.Module):
    """Feature-specific embedding tokenizer for mixed tabular tensors.

    Categorical feature value 0 is treated as the missing class. Numerical
    features use one value embedding and two validity embeddings per feature.
    """

    def __init__(
        self,
        *,
        role: str,
        n_num_features: int,
        category_sizes: Sequence[int],
        token_dim: int,
    ) -> None:
        super().__init__()
        role = role.strip()
        if not role:
            raise ValueError("role must not be blank.")
        if n_num_features < 0:
            raise ValueError("n_num_features must not be negative.")
        if token_dim <= 0:
            raise ValueError("token_dim must be positive.")
        self.role = role
        self.n_num_features = int(n_num_features)
        self.category_sizes = [int(size) for size in category_sizes]
        if any(size <= 0 for size in self.category_sizes):
            raise ValueError("All category sizes must be positive.")
        self.n_cat_features = len(self.category_sizes)
        self.embedding_dim = int(token_dim)
        self.num_value_embedding = nn.Parameter(
            torch.empty(self.n_num_features, self.embedding_dim)
        )
        self.num_valid_embedding = nn.Parameter(
            torch.empty(self.n_num_features, 2, self.embedding_dim)
        )
        self.cat_embeddings = nn.ModuleList(
            nn.Embedding(size, self.embedding_dim)
            for size in self.category_sizes
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.n_num_features > 0:
            nn.init.normal_(self.num_value_embedding, std=0.02)
            nn.init.normal_(self.num_valid_embedding, std=0.02)
        for embedding in self.cat_embeddings:
            nn.init.normal_(embedding.weight, std=0.02)

    @property
    def n_tokens(self) -> int:
        return self.n_num_features + self.n_cat_features

    def forward(
        self,
        num_features: Tensor | None,
        cat_features: Tensor | None,
        num_mask: Tensor | None,
        cat_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        batch_size = _infer_batch_size(num_features, cat_features, num_mask, cat_mask)
        device = _infer_device(num_features, cat_features, num_mask, cat_mask)
        token_groups: list[Tensor] = []

        if self.n_num_features > 0:
            num_values = _validate_matrix(
                num_features,
                f"{self.role}_num",
                self.n_num_features,
            ).to(device=device, dtype=torch.float32)
            num_mask_values = _validate_matrix(
                num_mask,
                f"{self.role}_num_mask",
                self.n_num_features,
            ).to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
            valid_indices = num_mask_values.to(dtype=torch.long).clamp(0, 1)
            value_tokens = num_values.unsqueeze(-1) * self.num_value_embedding.unsqueeze(0)
            feature_ids = torch.arange(self.n_num_features, device=device).view(1, -1)
            valid_tokens = self.num_valid_embedding[
                feature_ids,
                valid_indices,
            ]
            token_groups.append(value_tokens + valid_tokens)

        if self.n_cat_features > 0:
            cat_values = _validate_matrix(
                cat_features,
                f"{self.role}_cat",
                self.n_cat_features,
            ).to(device=device, dtype=torch.long)
            cat_mask_values = _validate_matrix(
                cat_mask,
                f"{self.role}_cat_mask",
                self.n_cat_features,
            ).to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
            for index, (size, embedding) in enumerate(
                zip(self.category_sizes, self.cat_embeddings)
            ):
                values = cat_values[:, index].clamp(min=0, max=size - 1)
                observed = cat_mask_values[:, index].to(dtype=torch.bool)
                values = torch.where(observed, values, torch.zeros_like(values))
                token_groups.append(embedding(values).unsqueeze(1))

        if not token_groups:
            raise ValueError(f"{self.role} must contain at least one feature.")
        tokens = torch.cat(token_groups, dim=1)
        masks = torch.ones(
            (batch_size, self.n_tokens),
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return tokens, masks


def _make_cvae_tokenizer_stack(
    *,
    tokenizer_type: str,
    role: str,
    n_num_features: int,
    category_sizes: Sequence[int],
    embedding_dim: int,
    token_dim: int,
) -> tuple[nn.Module, nn.Module]:
    if tokenizer_type == "eddi":
        tokenizer = make_tabular_feature_tokenizer(
            role=role,
            n_num_features=n_num_features,
            category_sizes=category_sizes,
            embedding_dim=embedding_dim,
        )
        token_network = _make_token_network(
            input_dim=embedding_dim,
            out_dim=token_dim,
        )
        return tokenizer, token_network
    if tokenizer_type in {"raw_embedding", "raw_linear"}:
        tokenizer = _RawEmbeddingFeatureTokenizer(
            role=role,
            n_num_features=n_num_features,
            category_sizes=category_sizes,
            token_dim=token_dim,
        )
        return tokenizer, nn.Identity()
    raise ValueError(f"Unsupported tokenizer_type: {tokenizer_type!r}")


def _raw_tabular_context(
    *,
    num_features: Tensor | None,
    cat_features: Tensor | None,
    num_mask: Tensor | None,
    cat_mask: Tensor | None,
    n_num_features: int,
    category_sizes: Sequence[int],
    name: str,
) -> Tensor:
    batch_size = _infer_batch_size(num_features, cat_features, num_mask, cat_mask)
    device = _infer_device(num_features, cat_features, num_mask, cat_mask)
    parts: list[Tensor] = []

    if n_num_features > 0:
        num_values = _validate_matrix(
            num_features,
            f"{name}_num",
            n_num_features,
        ).to(device=device, dtype=torch.float32)
        num_mask_values = _validate_matrix(
            num_mask,
            f"{name}_num_mask",
            n_num_features,
        ).to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        parts.extend([num_values * num_mask_values, num_mask_values])

    category_sizes = [int(size) for size in category_sizes]
    if category_sizes:
        cat_values = _validate_matrix(
            cat_features,
            f"{name}_cat",
            len(category_sizes),
        ).to(device=device, dtype=torch.long)
        cat_mask_values = _validate_matrix(
            cat_mask,
            f"{name}_cat_mask",
            len(category_sizes),
        ).to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        onehot_parts = []
        for index, size in enumerate(category_sizes):
            values = cat_values[:, index].clamp(min=0, max=size - 1)
            onehot = F.one_hot(values, num_classes=size).to(dtype=torch.float32)
            onehot_parts.append(onehot * cat_mask_values[:, index : index + 1])
        parts.extend([torch.cat(onehot_parts, dim=1), cat_mask_values])

    if not parts:
        return torch.empty((batch_size, 0), device=device, dtype=torch.float32)
    return torch.cat(parts, dim=1)


def _transformer_encoder_config(model_config: Mapping[str, object]) -> Mapping[str, object]:
    if "encoder" in model_config:
        encoder_config = _mapping_config(model_config, "encoder")
        encoder_type = str(encoder_config.get("type", "transformer")).lower()
        if encoder_type != "transformer":
            raise ValueError("model.encoder.type must be 'transformer'.")
        return encoder_config
    return _mapping_config(model_config, "transformer")


def _transformer_prior_config(model_config: Mapping[str, object]) -> Mapping[str, object] | None:
    if "prior" not in model_config:
        return None
    prior_config = _mapping_config(model_config, "prior")
    prior_type = str(prior_config.get("type", "transformer")).lower()
    if prior_type != "transformer":
        raise ValueError("model.prior.type must be 'transformer'.")
    return prior_config


def _normalize_layers_alias(
    config: Mapping[str, object],
    *,
    key_path: str,
) -> dict[str, object]:
    normalized = dict(config)
    if "layers" not in normalized:
        return normalized
    layers = normalized.pop("layers")
    if "hidden_layers" in normalized:
        hidden_layers = list(_sequence_config(normalized["hidden_layers"]))
        alias_layers = list(_sequence_config(layers))
        if hidden_layers != alias_layers:
            raise ValueError(f"{key_path}.layers and {key_path}.hidden_layers disagree.")
    normalized["hidden_layers"] = layers
    return normalized


def _build_source_encoder(
    *,
    n_num_features: int,
    category_sizes: Sequence[int],
    config: Mapping[str, object],
) -> nn.Module:
    encoder_type = str(config.get("type", "eddi")).lower()
    if encoder_type == "mlp":
        hidden_layers = config.get("hidden_layers", [])
        if not isinstance(hidden_layers, Sequence) or isinstance(hidden_layers, str):
            raise ValueError("source_encoder.hidden_layers must be a sequence.")
        return make_mlp_source_encoder(
            n_num_features=n_num_features,
            category_sizes=category_sizes,
            hidden_layers=hidden_layers,
            output_dim=int(config["output_dim"]),
            dropout=float(config.get("dropout", 0.0)),
            activation=str(config.get("activation", "relu")),
            normalization=str(config.get("normalization", "none")),
        )
    if encoder_type != "eddi":
        raise ValueError("source_encoder.type must be either 'eddi' or 'mlp'.")
    hidden_layers = config.get("hidden_layers", [])
    if not isinstance(hidden_layers, Sequence) or isinstance(hidden_layers, str):
        raise ValueError("source_encoder.hidden_layers must be a sequence.")
    embedding_dim = int(config.get("embedding_dim", 64))
    output_dim = int(config.get("output_dim", embedding_dim))
    return make_eddi_source_encoder(
        role="source",
        n_num_features=n_num_features,
        category_sizes=category_sizes,
        embedding_dim=embedding_dim,
        hidden_layers=hidden_layers,
        output_dim=output_dim,
        aggregation=str(config.get("aggregation", "sum")),
    )


def _build_eddi_source_encoder(
    *,
    n_num_features: int,
    category_sizes: Sequence[int],
    config: Mapping[str, object],
) -> nn.Module:
    return _build_source_encoder(
        n_num_features=n_num_features,
        category_sizes=category_sizes,
        config={**dict(config), "type": "eddi"},
    )


def _make_transformer_encoder(
    *,
    token_dim: int,
    config: Mapping[str, object],
    default_dropout: float,
) -> nn.TransformerEncoder:
    n_heads = int(config.get("n_heads", 4))
    n_layers = int(config.get("n_layers", 2))
    ff_dim = int(config.get("ff_dim", 4 * token_dim))
    dropout = float(config.get("dropout", default_dropout))
    activation = str(config.get("activation", "gelu"))
    norm_first = bool(config.get("norm_first", True))
    if token_dim <= 0:
        raise ValueError("token_dim must be positive.")
    if n_heads <= 0:
        raise ValueError("transformer.n_heads must be positive.")
    if token_dim % n_heads != 0:
        raise ValueError("model.d_model must be divisible by transformer.n_heads.")
    if n_layers <= 0:
        raise ValueError("transformer.n_layers must be positive.")
    if ff_dim <= 0:
        raise ValueError("transformer.ff_dim must be positive.")
    if not 0 <= dropout < 1:
        raise ValueError("transformer.dropout must be in [0, 1).")
    if activation not in {"relu", "gelu"}:
        raise ValueError("transformer.activation must be either 'relu' or 'gelu'.")

    layer = nn.TransformerEncoderLayer(
        d_model=token_dim,
        nhead=n_heads,
        dim_feedforward=ff_dim,
        dropout=dropout,
        activation=activation,
        batch_first=True,
        norm_first=norm_first,
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers)


def _make_token_network(
    *,
    input_dim: int,
    out_dim: int,
) -> nn.Module:
    input_dim = int(input_dim)
    out_dim = int(out_dim)
    if input_dim <= 0:
        raise ValueError("tokenizer.embedding_dim must be positive.")
    if out_dim <= 0:
        raise ValueError("tokenizer.out_dim must be positive.")
    if input_dim == out_dim:
        return nn.Identity()
    return nn.Linear(input_dim, out_dim)


def _make_mlp(
    *,
    input_dim: int,
    hidden_layers: Sequence[int],
    output_dim: int,
    dropout: float = 0.0,
    activation: str = "relu",
    batch_norm: bool = False,
    normalization: str | None = None,
) -> nn.Sequential:
    normalization = _resolve_mlp_normalization(
        batch_norm=batch_norm,
        normalization=normalization,
    )
    layers: list[nn.Module] = []
    current_dim = int(input_dim)
    for hidden_dim in hidden_layers:
        hidden_dim = int(hidden_dim)
        if hidden_dim <= 0:
            raise ValueError("Hidden layer dimensions must be positive.")
        layers.append(nn.Linear(current_dim, hidden_dim))
        if normalization == "batch_norm":
            layers.append(nn.BatchNorm1d(hidden_dim))
        elif normalization == "layer_norm":
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(_make_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, int(output_dim)))
    return nn.Sequential(*layers)


def _optional_normalization_config(config: Mapping[str, object]) -> str | None:
    if "normalization" not in config:
        return None
    value = config["normalization"]
    if value is None:
        return None
    return str(value)


def _resolve_mlp_normalization(
    *,
    batch_norm: bool = False,
    normalization: str | None = None,
) -> str:
    if normalization is None or normalization.strip() == "":
        return "batch_norm" if batch_norm else "none"
    normalized = normalization.strip().lower().replace("-", "_")
    aliases = {
        "batchnorm": "batch_norm",
        "bn": "batch_norm",
        "layernorm": "layer_norm",
        "ln": "layer_norm",
        "none": "none",
        "false": "none",
        "off": "none",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"none", "batch_norm", "layer_norm"}:
        raise ValueError(
            "normalization must be one of: none, batch_norm, layer_norm."
        )
    return normalized


def _make_activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def _mapping_config(config: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = config.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"model.{key} must be a mapping.")
    return value


def _sequence_config(value: object) -> Sequence[int]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError("Expected a sequence of hidden layer dimensions.")
    return value


def _optional_sequence_config(value: object) -> Sequence[int] | None:
    if value is None:
        return None
    return _sequence_config(value)


def _masked_feature_sum_mean(
    per_feature: Tensor,
    mask: Tensor,
    *,
    sample_weights: Tensor | None = None,
) -> Tensor:
    if per_feature.shape != mask.shape:
        raise ValueError(
            f"Mask shape must match loss shape: {tuple(mask.shape)} != "
            f"{tuple(per_feature.shape)}."
        )
    if sample_weights is not None:
        sample_weights = _sample_weight_vector(
            sample_weights,
            batch_size=per_feature.shape[0],
            device=per_feature.device,
            dtype=per_feature.dtype,
        )
        mask = mask * sample_weights.view(-1, 1)
    return (per_feature * mask).sum(dim=1).mean()


def _masked_feature_count(
    mask: Tensor | None,
    expected_features: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
    sample_weights: Tensor | None = None,
) -> Tensor:
    if expected_features == 0:
        return torch.zeros((), device=device, dtype=dtype)
    mask_values = (
        _validate_matrix(mask, name, expected_features)
        .to(device=device, dtype=dtype)
        .clamp(0.0, 1.0)
    )
    if sample_weights is not None:
        sample_weights = _sample_weight_vector(
            sample_weights,
            batch_size=mask_values.shape[0],
            device=device,
            dtype=dtype,
        )
        mask_values = mask_values * sample_weights.view(-1, 1)
    return mask_values.sum()


def _safe_divide(numerator: Tensor, denominator: Tensor) -> Tensor:
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1.0),
        torch.zeros_like(numerator),
    )


def _diagonal_gaussian_kl(
    mu: Tensor,
    logvar: Tensor,
    prior_mu: Tensor,
    prior_logvar: Tensor,
    *,
    sample_weights: Tensor | None = None,
) -> Tensor:
    if (
        mu.shape != logvar.shape
        or mu.shape != prior_mu.shape
        or mu.shape != prior_logvar.shape
    ):
        raise ValueError("Posterior and prior parameters must have matching shapes.")
    var_ratio = (logvar - prior_logvar).exp()
    mean_difference = (mu - prior_mu).pow(2) * (-prior_logvar).exp()
    per_sample = 0.5 * (
        prior_logvar
        - logvar
        + var_ratio
        + mean_difference
        - 1.0
    ).sum(dim=1)
    if sample_weights is not None:
        sample_weights = _sample_weight_vector(
            sample_weights,
            batch_size=per_sample.shape[0],
            device=per_sample.device,
            dtype=per_sample.dtype,
        )
        per_sample = per_sample * sample_weights
    return per_sample.mean()


def _sample_weight_vector(
    weights: Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if weights is None:
        return torch.ones((batch_size,), device=device, dtype=dtype)
    weights = weights.to(device=device, dtype=dtype)
    if weights.ndim == 2 and weights.shape[1] == 1:
        weights = weights.squeeze(1)
    if weights.ndim != 1 or weights.shape[0] != batch_size:
        raise ValueError(
            f"sample_loss_weights must have shape ({batch_size},) or "
            f"({batch_size}, 1), got {tuple(weights.shape)}."
        )
    if torch.any(~torch.isfinite(weights)) or torch.any(weights < 0):
        raise ValueError("sample_loss_weights must be non-negative and finite.")
    return weights


def _validate_matrix(
    values: Tensor | None,
    name: str,
    expected_features: int,
) -> Tensor:
    if values is None:
        raise ValueError(f"{name} must not be None when its features exist.")
    if values.ndim != 2:
        raise ValueError(f"{name} must have shape (batch, features).")
    if values.shape[1] != expected_features:
        raise ValueError(
            f"Expected {name} to have {expected_features} features, "
            f"got {values.shape[1]}."
        )
    return values


def _infer_batch_size(*values: Tensor | None) -> int:
    batch_sizes = [int(value.shape[0]) for value in values if value is not None]
    if not batch_sizes:
        raise ValueError("At least one input tensor must be provided.")
    if len(set(batch_sizes)) != 1:
        raise ValueError(f"Input tensors have inconsistent batch sizes: {batch_sizes}")
    return batch_sizes[0]


def _infer_device(*values: Tensor | None) -> torch.device:
    for value in values:
        if value is not None:
            return value.device
    return torch.device("cpu")
