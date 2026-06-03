"""Conditional Gaussian + multinomial diffusion for tabular targets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from diffusion.schedules import get_named_beta_schedule
from diffusion.utils import (
    LOG_EPS,
    extract,
    log_1_min_a,
    log_add_exp,
    normal_kl,
)


class ConditionalGaussianMultinomialDiffusion(nn.Module):
    """Diffuse target values conditioned on a separate source representation.

    The model output contains target values only:

    ``target_num | target_cat_logits``

    Target masks can be provided to weight training losses, but masks are not
    diffused, denoised, sampled, or returned as model outputs.
    """

    def __init__(
        self,
        *,
        target_n_num_features: int,
        target_category_sizes: list[int] | np.ndarray,
        denoise_fn: nn.Module,
        num_timesteps: int = 1000,
        scheduler: str = "cosine",
        gaussian_loss_type: str = "mse",
        gaussian_parametrization: str = "eps",
        multinomial_loss_type: str = "vb_stochastic",
        categorical_parametrization: str = "x0",
    ) -> None:
        super().__init__()
        if target_n_num_features < 0:
            raise ValueError("target_n_num_features must not be negative.")
        if num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive.")
        if gaussian_loss_type not in {"mse", "kl"}:
            raise ValueError("gaussian_loss_type must be 'mse' or 'kl'.")
        if gaussian_parametrization not in {"eps", "x0"}:
            raise ValueError("gaussian_parametrization must be 'eps' or 'x0'.")
        if multinomial_loss_type not in {"vb_stochastic", "vb_all"}:
            raise ValueError("multinomial_loss_type must be 'vb_stochastic' or 'vb_all'.")
        if categorical_parametrization not in {"x0", "direct"}:
            raise ValueError("categorical_parametrization must be 'x0' or 'direct'.")

        category_sizes = np.asarray(target_category_sizes, dtype=np.int64)
        if category_sizes.ndim != 1:
            raise ValueError("target_category_sizes must be one-dimensional.")
        if np.any(category_sizes <= 0):
            raise ValueError("All target category sizes must be positive.")

        self.target_n_num_features = int(target_n_num_features)
        self.target_category_sizes = category_sizes
        self.target_n_cat_features = int(len(category_sizes))
        self.target_cat_dim = int(category_sizes.sum())
        self.target_dim = self.target_n_num_features + self.target_cat_dim
        if self.target_dim <= 0:
            raise ValueError("At least one numerical or categorical target is required.")

        self.denoise_fn = denoise_fn
        self.num_timesteps = int(num_timesteps)
        self.gaussian_loss_type = gaussian_loss_type
        self.gaussian_parametrization = gaussian_parametrization
        self.multinomial_loss_type = multinomial_loss_type
        self.categorical_parametrization = categorical_parametrization

        betas = get_named_beta_schedule(scheduler, self.num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        alphas_cumprod_next = np.append(alphas_cumprod[1:], 0.0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_log_variance_clipped = np.log(
            np.append(posterior_variance[1], posterior_variance[1:])
        )
        posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))
        self.register_buffer("alphas", torch.tensor(alphas, dtype=torch.float32))
        self.register_buffer(
            "alphas_cumprod",
            torch.tensor(alphas_cumprod, dtype=torch.float32),
        )
        self.register_buffer(
            "alphas_cumprod_prev",
            torch.tensor(alphas_cumprod_prev, dtype=torch.float32),
        )
        self.register_buffer(
            "alphas_cumprod_next",
            torch.tensor(alphas_cumprod_next, dtype=torch.float32),
        )
        self.register_buffer(
            "sqrt_alphas_cumprod",
            torch.tensor(np.sqrt(alphas_cumprod), dtype=torch.float32),
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.tensor(np.sqrt(1.0 - alphas_cumprod), dtype=torch.float32),
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod",
            torch.tensor(np.sqrt(1.0 / alphas_cumprod), dtype=torch.float32),
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.tensor(np.sqrt(1.0 / alphas_cumprod - 1.0), dtype=torch.float32),
        )
        self.register_buffer(
            "posterior_variance",
            torch.tensor(posterior_variance, dtype=torch.float32),
        )
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.tensor(posterior_log_variance_clipped, dtype=torch.float32),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            torch.tensor(posterior_mean_coef1, dtype=torch.float32),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            torch.tensor(posterior_mean_coef2, dtype=torch.float32),
        )

        log_alpha = torch.log(self.alphas)
        log_cumprod_alpha = torch.log(self.alphas_cumprod)
        self.register_buffer("log_alpha", log_alpha)
        self.register_buffer("log_1_min_alpha", log_1_min_a(log_alpha))
        self.register_buffer("log_cumprod_alpha", log_cumprod_alpha)
        self.register_buffer(
            "log_1_min_cumprod_alpha",
            log_1_min_a(log_cumprod_alpha),
        )

        self._cat_group_specs: tuple[tuple[str, int], ...] = ()
        if self.target_n_cat_features > 0:
            expanded = np.concatenate(
                [np.repeat(size, size) for size in self.target_category_sizes]
            )
            offsets = np.append([0], np.cumsum(self.target_category_sizes))
            feature_index_expanded = np.concatenate(
                [
                    np.full(int(size), feature_index, dtype=np.int64)
                    for feature_index, size in enumerate(self.target_category_sizes)
                ]
            )
        else:
            expanded = np.array([], dtype=np.int64)
            offsets = np.array([0], dtype=np.int64)
            feature_index_expanded = np.array([], dtype=np.int64)
        self.register_buffer(
            "cat_class_sizes",
            torch.tensor(self.target_category_sizes, dtype=torch.long),
        )
        self.register_buffer(
            "num_classes_expanded",
            torch.tensor(expanded, dtype=torch.float32),
        )
        self.register_buffer(
            "cat_feature_index_expanded",
            torch.tensor(feature_index_expanded, dtype=torch.long),
        )
        self.register_buffer("cat_offsets", torch.tensor(offsets, dtype=torch.long))
        self._register_categorical_groups(offsets)

    def _register_categorical_groups(self, offsets: np.ndarray) -> None:
        """Build one padded categorical group for vectorized operations."""
        if self.target_n_cat_features == 0:
            return

        bucket_size = int(self.target_category_sizes.max())
        if bucket_size > 64:
            raise ValueError(
                "Categorical grouping currently supports up to 64 classes per feature."
            )
        feature_indices = np.arange(self.target_n_cat_features, dtype=np.int64)
        flat_indices = np.zeros(
            (self.target_n_cat_features, bucket_size),
            dtype=np.int64,
        )
        class_mask = np.zeros(
            (self.target_n_cat_features, bucket_size),
            dtype=np.bool_,
        )
        for row, size in enumerate(self.target_category_sizes.tolist()):
            start = int(offsets[row])
            flat_indices[row, : int(size)] = np.arange(start, start + int(size))
            class_mask[row, : int(size)] = True

        group_name = "cat_group_all"
        self.register_buffer(
            f"{group_name}_feature_indices",
            torch.tensor(feature_indices, dtype=torch.long),
        )
        self.register_buffer(
            f"{group_name}_flat_indices",
            torch.tensor(flat_indices, dtype=torch.long),
        )
        self.register_buffer(
            f"{group_name}_class_mask",
            torch.tensor(class_mask, dtype=torch.bool),
        )
        self._cat_group_specs = ((group_name, bucket_size),)

    @staticmethod
    def _bucket_size(size: int) -> int:
        if size <= 4:
            return 4
        if size <= 8:
            return 8
        if size <= 16:
            return 16
        if size <= 32:
            return 32
        if size <= 64:
            return 64
        raise ValueError(
            "Categorical grouping currently supports up to 64 classes per feature."
        )

    def _iter_cat_groups(self):
        for group_name, bucket_size in self._cat_group_specs:
            yield (
                bucket_size,
                getattr(self, f"{group_name}_feature_indices"),
                getattr(self, f"{group_name}_flat_indices"),
                getattr(self, f"{group_name}_class_mask"),
            )

    @staticmethod
    def _gather_cat_group(
        values: Tensor,
        flat_indices: Tensor,
        class_mask: Tensor,
        *,
        fill_value: float,
    ) -> Tensor:
        batch_size = int(values.shape[0])
        grouped = values.index_select(1, flat_indices.reshape(-1)).reshape(
            batch_size,
            int(flat_indices.shape[0]),
            int(flat_indices.shape[1]),
        )
        return torch.where(
            class_mask.unsqueeze(0),
            grouped,
            grouped.new_full((), fill_value),
        )

    @staticmethod
    def _scatter_cat_group(
        output: Tensor,
        grouped_values: Tensor,
        flat_indices: Tensor,
        class_mask: Tensor,
    ) -> None:
        output[:, flat_indices[class_mask]] = grouped_values[:, class_mask].to(
            dtype=output.dtype
        )

    def _index_to_log_onehot(self, indices: Tensor) -> Tensor:
        if indices.ndim != 2:
            raise ValueError("target_cat must have shape (batch, n_cat).")
        if indices.shape[1] != self.target_n_cat_features:
            raise ValueError(
                f"Expected {self.target_n_cat_features} categorical features, "
                f"got {indices.shape[1]}."
            )

        output = torch.empty(
            (indices.shape[0], self.target_cat_dim),
            device=indices.device,
            dtype=self.betas.dtype,
        )
        for bucket_size, feature_indices, flat_indices, class_mask in self._iter_cat_groups():
            values = indices.index_select(1, feature_indices).long()
            onehot = F.one_hot(values, num_classes=bucket_size).to(dtype=output.dtype)
            log_onehot = torch.log(onehot.clamp(min=LOG_EPS))
            self._scatter_cat_group(output, log_onehot, flat_indices, class_mask)
        return output

    def _log_onehot_to_index(self, log_x: Tensor) -> Tensor:
        output = torch.empty(
            (log_x.shape[0], self.target_n_cat_features),
            device=log_x.device,
            dtype=torch.long,
        )
        for _, feature_indices, flat_indices, class_mask in self._iter_cat_groups():
            grouped = self._gather_cat_group(
                log_x,
                flat_indices,
                class_mask,
                fill_value=float("-inf"),
            )
            output[:, feature_indices] = grouped.argmax(dim=2)
        return output

    def _categorical_log_softmax(self, logits: Tensor) -> Tensor:
        output = torch.empty_like(logits)
        for _, _, flat_indices, class_mask in self._iter_cat_groups():
            grouped = self._gather_cat_group(
                logits,
                flat_indices,
                class_mask,
                fill_value=float("-inf"),
            )
            log_probs = torch.log_softmax(grouped, dim=2)
            self._scatter_cat_group(output, log_probs, flat_indices, class_mask)
        return output

    def _categorical_sliced_logsumexp(self, values: Tensor) -> Tensor:
        output = torch.empty_like(values)
        for _, _, flat_indices, class_mask in self._iter_cat_groups():
            grouped = self._gather_cat_group(
                values,
                flat_indices,
                class_mask,
                fill_value=float("-inf"),
            )
            logsumexp = grouped.logsumexp(dim=2, keepdim=True).expand_as(grouped)
            self._scatter_cat_group(output, logsumexp, flat_indices, class_mask)
        return output

    def mixed_loss(
        self,
        *,
        target_num: Tensor | None,
        target_cat: Tensor | None,
        condition: Tensor | None = None,
        target_num_mask: Tensor | None = None,
        target_cat_mask: Tensor | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> dict[str, Tensor]:
        """Compute one stochastic training loss step."""
        batch_size = _infer_target_batch_size(target_num, target_cat)
        device = _infer_target_device(target_num, target_cat)
        timesteps = torch.randint(
            0,
            self.num_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )

        noisy_num = self._empty_num(batch_size, device)
        gaussian_noise = self._empty_num(batch_size, device)
        if self.target_n_num_features > 0:
            target_num = self._validate_num(target_num).to(dtype=torch.float32)
            gaussian_noise = torch.randn_like(target_num)
            noisy_num = self.gaussian_q_sample(target_num, timesteps, gaussian_noise)

        noisy_cat = self._empty_cat(batch_size, device)
        log_cat_start = self._empty_cat(batch_size, device)
        if self.target_n_cat_features > 0:
            if target_cat_mask is not None and target_cat is not None:
                invalid_missing = (target_cat == 0) & target_cat_mask.to(dtype=torch.bool)
                if invalid_missing.any():
                    raise ValueError(
                        "target_cat contains missing class 0 values marked valid by target_cat_mask."
                    )
            target_cat = self._safe_categorical_targets(target_cat)
            log_cat_start = self._index_to_log_onehot(target_cat).to(device=device)
            noisy_cat = self.multinomial_q_sample(log_cat_start, timesteps)
            if target_cat_mask is not None:
                noisy_cat = self._apply_missing_categorical_uniform(
                    noisy_cat,
                    target_cat_mask,
                )

        model_input = torch.cat([noisy_num, noisy_cat], dim=1)
        denoise_model_kwargs = dict(model_kwargs or {})
        if getattr(self.denoise_fn, "uses_target_masks", False):
            if target_num_mask is not None:
                denoise_model_kwargs["target_num_mask"] = target_num_mask
            if target_cat_mask is not None:
                denoise_model_kwargs["target_cat_mask"] = target_cat_mask

        model_out = self._call_denoise_fn(
            model_input,
            timesteps,
            condition=condition,
            model_kwargs=denoise_model_kwargs,
        )
        if model_out.shape != model_input.shape:
            raise ValueError(
                f"denoise_fn output must have shape {tuple(model_input.shape)}, "
                f"got {tuple(model_out.shape)}."
            )

        model_out_num = model_out[:, : self.target_n_num_features]
        model_out_cat = model_out[:, self.target_n_num_features :]

        loss_gaussian = model_out.new_zeros(())
        if self.target_n_num_features > 0:
            loss_gaussian = self._gaussian_loss(
                model_out_num,
                target_num,
                noisy_num,
                timesteps,
                gaussian_noise,
                target_num_mask,
            )

        loss_multinomial = model_out.new_zeros(())
        if self.target_n_cat_features > 0:
            loss_multinomial = self._multinomial_loss(
                model_out_cat,
                log_cat_start,
                noisy_cat,
                timesteps,
                target_cat_mask,
            )

        loss = loss_gaussian + loss_multinomial
        return {
            "loss": loss,
            "loss_gaussian": loss_gaussian,
            "loss_multinomial": loss_multinomial,
        }

    def gaussian_q_sample(
        self,
        x_start: Tensor,
        timesteps: Tensor,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Sample q(x_t | x_0) for numerical targets."""
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape)
            * noise
        )

    def multinomial_q_sample(self, log_x_start: Tensor, timesteps: Tensor) -> Tensor:
        """Sample q(x_t | x_0) for categorical targets in log one-hot form."""
        log_probs = self.multinomial_q_pred(log_x_start, timesteps)
        return self.log_sample_categorical(log_probs)

    def multinomial_q_pred(self, log_x_start: Tensor, timesteps: Tensor) -> Tensor:
        log_cumprod_alpha_t = extract(
            self.log_cumprod_alpha,
            timesteps,
            log_x_start.shape,
        )
        log_1_min_cumprod_alpha = extract(
            self.log_1_min_cumprod_alpha,
            timesteps,
            log_x_start.shape,
        )
        return log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - torch.log(self.num_classes_expanded),
        )

    def multinomial_q_pred_one_timestep(
        self,
        log_x_t: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        log_alpha_t = extract(self.log_alpha, timesteps, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, timesteps, log_x_t.shape)
        return log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - torch.log(self.num_classes_expanded),
        )

    @torch.no_grad()
    def sample(
        self,
        *,
        batch_size: int,
        condition: Tensor | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> dict[str, Tensor]:
        """Sample target numerical values and categorical ids only."""
        device = self.betas.device
        num = self._empty_num(batch_size, device)
        if self.target_n_num_features > 0:
            num = torch.randn(
                (batch_size, self.target_n_num_features),
                device=device,
            )

        log_cat = self._empty_cat(batch_size, device)
        if self.target_n_cat_features > 0:
            uniform_logits = torch.zeros(
                (batch_size, self.target_cat_dim),
                device=device,
            )
            log_cat = self.log_sample_categorical(uniform_logits)

        for step in reversed(range(self.num_timesteps)):
            timesteps = torch.full(
                (batch_size,),
                step,
                device=device,
                dtype=torch.long,
            )
            model_input = torch.cat([num, log_cat], dim=1)
            model_out = self._call_denoise_fn(
                model_input,
                timesteps,
                condition=condition,
                model_kwargs=model_kwargs,
            )
            model_out_num = model_out[:, : self.target_n_num_features]
            model_out_cat = model_out[:, self.target_n_num_features :]
            if self.target_n_num_features > 0:
                num = self.gaussian_p_sample(model_out_num, num, timesteps)
            if self.target_n_cat_features > 0:
                log_cat = self.multinomial_p_sample(model_out_cat, log_cat, timesteps)

        output = {"num": num}
        if self.target_n_cat_features > 0:
            output["cat"] = self._log_onehot_to_observed_index(log_cat)
        else:
            output["cat"] = torch.empty((batch_size, 0), device=device, dtype=torch.long)
        return output

    def gaussian_p_sample(
        self,
        model_out: Tensor,
        x_t: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        out = self.gaussian_p_mean_variance(model_out, x_t, timesteps)
        noise = torch.randn_like(x_t)
        nonzero_mask = (timesteps != 0).float().view(-1, 1)
        return out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise

    def gaussian_p_mean_variance(
        self,
        model_out: Tensor,
        x_t: Tensor,
        timesteps: Tensor,
    ) -> dict[str, Tensor]:
        model_variance = torch.cat(
            [self.posterior_variance[1].unsqueeze(0), (1.0 - self.alphas)[1:]],
            dim=0,
        )
        model_log_variance = torch.log(model_variance)
        model_variance = extract(model_variance, timesteps, x_t.shape)
        model_log_variance = extract(model_log_variance, timesteps, x_t.shape)

        if self.gaussian_parametrization == "eps":
            pred_xstart = self._predict_xstart_from_eps(x_t, timesteps, model_out)
        elif self.gaussian_parametrization == "x0":
            pred_xstart = model_out
        else:
            raise NotImplementedError(self.gaussian_parametrization)

        model_mean, _, _ = self.gaussian_q_posterior_mean_variance(
            pred_xstart,
            x_t,
            timesteps,
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def gaussian_q_posterior_mean_variance(
        self,
        x_start: Tensor,
        x_t: Tensor,
        timesteps: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        posterior_mean = (
            extract(self.posterior_mean_coef1, timesteps, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, timesteps, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, timesteps, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped,
            timesteps,
            x_t.shape,
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def multinomial_p_sample(
        self,
        model_out: Tensor,
        log_x_t: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        return self.log_sample_categorical(
            self.multinomial_p_pred(model_out, log_x_t, timesteps)
        )

    def multinomial_p_pred(
        self,
        model_out: Tensor,
        log_x_t: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        if self.categorical_parametrization == "x0":
            log_x_recon = self.predict_categorical_start(model_out)
            return self.multinomial_q_posterior(log_x_recon, log_x_t, timesteps)
        if self.categorical_parametrization == "direct":
            return self.predict_categorical_start(model_out)
        raise NotImplementedError(self.categorical_parametrization)

    def predict_categorical_start(self, model_out: Tensor) -> Tensor:
        return self._categorical_log_softmax(model_out)

    def multinomial_q_posterior(
        self,
        log_x_start: Tensor,
        log_x_t: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        t_minus_1 = torch.where(
            timesteps > 0,
            timesteps - 1,
            torch.zeros_like(timesteps),
        )
        log_qxtmin_x0 = self.multinomial_q_pred(log_x_start, t_minus_1)
        log_qxtmin_x0 = torch.where(
            timesteps.view(-1, 1) == 0,
            log_x_start,
            log_qxtmin_x0,
        )
        unnormalized = log_qxtmin_x0 + self.multinomial_q_pred_one_timestep(
            log_x_t,
            timesteps,
        )
        return unnormalized - self._categorical_sliced_logsumexp(unnormalized)

    def log_sample_categorical(self, logits: Tensor) -> Tensor:
        output = torch.empty_like(logits)
        for bucket_size, _, flat_indices, class_mask in self._iter_cat_groups():
            grouped_logits = self._gather_cat_group(
                logits,
                flat_indices,
                class_mask,
                fill_value=float("-inf"),
            )
            uniform = torch.rand_like(grouped_logits)
            gumbel = -torch.log(-torch.log(uniform + LOG_EPS) + LOG_EPS)
            sampled = (grouped_logits + gumbel).argmax(dim=2)
            onehot = F.one_hot(sampled, num_classes=bucket_size).to(dtype=logits.dtype)
            log_onehot = torch.log(onehot.clamp(min=LOG_EPS))
            self._scatter_cat_group(output, log_onehot, flat_indices, class_mask)
        return output

    def _gaussian_loss(
        self,
        model_out: Tensor,
        x_start: Tensor,
        x_t: Tensor,
        timesteps: Tensor,
        noise: Tensor,
        mask: Tensor | None,
    ) -> Tensor:
        if self.gaussian_loss_type == "mse":
            per_feature = (noise - model_out) ** 2
        elif self.gaussian_loss_type == "kl":
            per_feature = self._gaussian_vb_terms(model_out, x_start, x_t, timesteps)
        else:
            raise NotImplementedError(self.gaussian_loss_type)
        return _masked_feature_mean(per_feature, mask)

    def _gaussian_vb_terms(
        self,
        model_out: Tensor,
        x_start: Tensor,
        x_t: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        true_mean, _, true_log_variance = self.gaussian_q_posterior_mean_variance(
            x_start,
            x_t,
            timesteps,
        )
        out = self.gaussian_p_mean_variance(model_out, x_t, timesteps)
        return normal_kl(
            true_mean,
            true_log_variance,
            out["mean"],
            out["log_variance"],
        )

    def _multinomial_loss(
        self,
        model_out: Tensor,
        log_x_start: Tensor,
        log_x_t: Tensor,
        timesteps: Tensor,
        mask: Tensor | None,
    ) -> Tensor:
        if self.multinomial_loss_type != "vb_stochastic":
            raise NotImplementedError("Only vb_stochastic is implemented for masked loss.")

        log_true = self.multinomial_q_posterior(log_x_start, log_x_t, timesteps)
        log_model = self.multinomial_p_pred(model_out, log_x_t, timesteps)

        per_feature = torch.empty(
            (model_out.shape[0], self.target_n_cat_features),
            device=model_out.device,
            dtype=torch.float32,
        )
        for _, feature_indices, flat_indices, class_mask in self._iter_cat_groups():
            valid = class_mask.unsqueeze(0)
            true_group = self._gather_cat_group(
                log_true,
                flat_indices,
                class_mask,
                fill_value=0.0,
            )
            model_group = self._gather_cat_group(
                log_model,
                flat_indices,
                class_mask,
                fill_value=0.0,
            )
            start_group = self._gather_cat_group(
                log_x_start,
                flat_indices,
                class_mask,
                fill_value=0.0,
            )
            true_prob = true_group.exp().masked_fill(~valid, 0.0)
            start_prob = start_group.exp().masked_fill(~valid, 0.0)
            log_diff = (true_group - model_group).masked_fill(~valid, 0.0)
            model_group = model_group.masked_fill(~valid, 0.0)
            kl = (true_prob * log_diff).sum(dim=2)
            decoder_nll = -(start_prob * model_group).sum(dim=2)
            per_feature[:, feature_indices] = torch.where(
                timesteps.view(-1, 1) == 0,
                decoder_nll,
                kl,
            )

        # Reference TabDDPM estimates the summed categorical VB term by dividing
        # the sampled-timestep loss by p(t). We sample t uniformly, so p(t)=1/T.
        timestep_prob = per_feature.new_full(
            (timesteps.shape[0], 1),
            1.0 / float(self.num_timesteps),
        )
        per_feature = per_feature / timestep_prob
        return _masked_feature_mean(per_feature, mask)

    def _predict_xstart_from_eps(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        eps: Tensor,
    ) -> Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape) * eps
        )

    def _safe_categorical_targets(self, target_cat: Tensor | None) -> Tensor:
        if target_cat is None:
            raise ValueError("target_cat must not be None when categorical targets exist.")
        if target_cat.ndim != 2:
            raise ValueError("target_cat must have shape (batch, n_cat).")
        if target_cat.shape[1] != self.target_n_cat_features:
            raise ValueError(
                f"Expected {self.target_n_cat_features} categorical features, "
                f"got {target_cat.shape[1]}."
            )
        safe = target_cat.long()
        if torch.any(safe < 0):
            raise ValueError("target_cat contains negative categorical ids.")
        if torch.any(safe >= self.cat_class_sizes.view(1, -1)):
            raise ValueError("target_cat contains values outside configured class ranges.")
        return safe

    def _log_onehot_to_observed_index(self, log_x: Tensor) -> Tensor:
        output = torch.empty(
            (log_x.shape[0], self.target_n_cat_features),
            device=log_x.device,
            dtype=torch.long,
        )
        for _, feature_indices, flat_indices, class_mask in self._iter_cat_groups():
            grouped = self._gather_cat_group(
                log_x,
                flat_indices,
                class_mask,
                fill_value=float("-inf"),
            )
            grouped = grouped.clone()
            grouped[:, :, 0] = float("-inf")
            sampled = grouped.argmax(dim=2)
            output[:, feature_indices] = torch.where(
                sampled > 0,
                sampled,
                torch.ones_like(sampled),
            )
        return output

    def _validate_num(self, target_num: Tensor | None) -> Tensor:
        if target_num is None:
            raise ValueError("target_num must not be None when numerical targets exist.")
        if target_num.ndim != 2:
            raise ValueError("target_num must have shape (batch, n_num).")
        if target_num.shape[1] != self.target_n_num_features:
            raise ValueError(
                f"Expected {self.target_n_num_features} numerical features, "
                f"got {target_num.shape[1]}."
            )
        return target_num

    def _apply_missing_categorical_uniform(
        self,
        log_cat: Tensor,
        cat_mask: Tensor,
    ) -> Tensor:
        cat_mask = cat_mask.to(device=log_cat.device, dtype=torch.bool)
        if cat_mask.ndim != 2 or cat_mask.shape[1] != self.target_n_cat_features:
            raise ValueError(
                "target_cat_mask must have shape (batch, target_n_cat_features)."
            )
        class_missing = ~cat_mask.index_select(1, self.cat_feature_index_expanded)
        uniform_log_prob = -torch.log(
            self.num_classes_expanded.to(device=log_cat.device, dtype=log_cat.dtype)
        )
        return torch.where(class_missing, uniform_log_prob.view(1, -1), log_cat)

    def _call_denoise_fn(
        self,
        model_input: Tensor,
        timesteps: Tensor,
        *,
        condition: Tensor | None,
        model_kwargs: Mapping[str, Any] | None,
    ) -> Tensor:
        kwargs = dict(model_kwargs or {})
        if condition is not None:
            kwargs["condition"] = condition
        return self.denoise_fn(model_input, timesteps, **kwargs)

    def _empty_num(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.empty(
            (batch_size, self.target_n_num_features),
            device=device,
            dtype=torch.float32,
        )

    def _empty_cat(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.empty(
            (batch_size, self.target_cat_dim),
            device=device,
            dtype=torch.float32,
        )


def _masked_feature_mean(per_feature: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return per_feature.mean()
    mask = mask.to(device=per_feature.device, dtype=per_feature.dtype)
    if mask.shape != per_feature.shape:
        raise ValueError(
            f"Mask shape must match loss shape: {tuple(mask.shape)} != "
            f"{tuple(per_feature.shape)}."
        )
    per_sample = (per_feature * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    valid_rows = mask.sum(dim=1) > 0
    if not valid_rows.any():
        return per_feature.new_zeros(())
    return per_sample[valid_rows].mean()


def _infer_target_batch_size(target_num: Tensor | None, target_cat: Tensor | None) -> int:
    sizes = []
    if target_num is not None:
        sizes.append(int(target_num.shape[0]))
    if target_cat is not None:
        sizes.append(int(target_cat.shape[0]))
    if not sizes:
        raise ValueError("At least one target tensor must be provided.")
    if len(set(sizes)) != 1:
        raise ValueError(f"Target tensors have inconsistent batch sizes: {sizes}")
    return sizes[0]


def _infer_target_device(target_num: Tensor | None, target_cat: Tensor | None) -> torch.device:
    if target_num is not None:
        return target_num.device
    if target_cat is not None:
        return target_cat.device
    return torch.device("cpu")
