"""Tabular feature tokenization and EDDI source embedding."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TabularFeatureTokenizer(nn.Module):
    """Convert mixed tabular features into per-feature embedding tokens.

    Each observed feature becomes a token carrying feature identity and value
    information:

    - Numerical features use ``value * feature_embedding + mask * feature_bias``.
    - Categorical features use ``feature_embedding + class_embedding``.
    - Missing features are returned as zero vectors.
    """

    def __init__(
        self,
        *,
        role: str,
        n_num_features: int,
        category_sizes: Sequence[int],
        embedding_dim: int,
    ) -> None:
        super().__init__()
        role = role.strip()
        if not role:
            raise ValueError("role must not be blank.")
        if n_num_features < 0:
            raise ValueError("n_num_features must not be negative.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        self.role = role
        self.n_num_features = int(n_num_features)
        self.category_sizes = [int(size) for size in category_sizes]
        if any(size <= 0 for size in self.category_sizes):
            raise ValueError("All category sizes must be positive.")
        self.n_cat_features = len(self.category_sizes)
        self.embedding_dim = int(embedding_dim)

        if self.n_num_features > 0:
            self.num_feature_embedding = nn.Embedding(
                self.n_num_features,
                self.embedding_dim,
            )
            self.num_feature_bias = nn.Parameter(
                torch.empty(self.n_num_features, self.embedding_dim)
            )
        else:
            self.num_feature_embedding = None
            self.register_parameter("num_feature_bias", None)

        self.max_n_classes = max(self.category_sizes, default=0)
        if self.n_cat_features > 0:
            self.cat_feature_embedding = nn.Embedding(
                self.n_cat_features,
                self.embedding_dim,
            )
            self.cat_value_embedding = nn.Parameter(
                torch.empty(
                    self.n_cat_features,
                    self.max_n_classes,
                    self.embedding_dim,
                )
            )
            self.register_buffer(
                "category_sizes_tensor",
                torch.tensor(self.category_sizes, dtype=torch.long),
            )
        else:
            self.cat_feature_embedding = None
            self.cat_value_embedding = None
            self.register_buffer(
                "category_sizes_tensor",
                torch.empty((0,), dtype=torch.long),
            )

        self.reset_parameters()

    @property
    def n_tokens(self) -> int:
        return self.n_num_features + self.n_cat_features

    def reset_parameters(self) -> None:
        if self.num_feature_embedding is not None:
            nn.init.normal_(self.num_feature_embedding.weight, std=0.02)
        if self.num_feature_bias is not None:
            nn.init.normal_(self.num_feature_bias, std=0.02)
        if self.cat_feature_embedding is not None:
            nn.init.normal_(self.cat_feature_embedding.weight, std=0.02)
        if self.cat_value_embedding is not None:
            nn.init.normal_(self.cat_value_embedding, std=0.02)

    def forward(
        self,
        num_features: Tensor | None,
        cat_features: Tensor | None,
        num_mask: Tensor | None,
        cat_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        return self.tokenize(
            num_features=num_features,
            cat_features=cat_features,
            num_mask=num_mask,
            cat_mask=cat_mask,
        )

    def tokenize(
        self,
        *,
        num_features: Tensor | None,
        cat_features: Tensor | None,
        num_mask: Tensor | None,
        cat_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Return raw feature tokens and their observation mask.

        Returns:
            tokens: ``(batch, n_tokens, embedding_dim)``.
            masks: ``(batch, n_tokens)`` with 1 for observed tokens and 0 otherwise.

        Missing tokens are zeroed before returning. Downstream transformer
        backbones may feed those zero tokens directly into self-attention.
        """
        if self.n_num_features == 0 and self.n_cat_features == 0:
            raise ValueError("At least one numerical or categorical feature is required.")

        _infer_batch_size(num_features, cat_features, num_mask, cat_mask)
        device = _infer_device(num_features, cat_features, num_mask, cat_mask)
        token_groups = []
        mask_groups = []

        if self.n_num_features > 0:
            num_values = self._validate_role_matrix(
                num_features,
                name="num_features",
                expected_features=self.n_num_features,
            ).to(device=device, dtype=torch.float32)
            num_mask_values = self._validate_role_matrix(
                num_mask,
                name="num_mask",
                expected_features=self.n_num_features,
            ).to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
            feature_ids = torch.arange(self.n_num_features, device=device)
            feature_tokens = self.num_feature_embedding(feature_ids)
            feature_bias = self.num_feature_bias[feature_ids]
            num_tokens = num_values.unsqueeze(-1) * feature_tokens.unsqueeze(0)
            num_tokens = (
                num_tokens
                + num_mask_values.unsqueeze(-1) * feature_bias.unsqueeze(0)
            )
            token_groups.append(num_tokens)
            mask_groups.append(num_mask_values)

        if self.n_cat_features > 0:
            cat_values = self._validate_role_matrix(
                cat_features,
                name="cat_features",
                expected_features=self.n_cat_features,
            ).to(device=device, dtype=torch.long)
            cat_mask_values = self._validate_role_matrix(
                cat_mask,
                name="cat_mask",
                expected_features=self.n_cat_features,
            ).to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
            self._validate_cat_range(cat_values)
            feature_ids = torch.arange(self.n_cat_features, device=device)
            feature_tokens = self.cat_feature_embedding(feature_ids)
            cat_value_tokens = self.cat_value_embedding[
                feature_ids.unsqueeze(0),
                cat_values,
            ]
            cat_tokens = feature_tokens.unsqueeze(0) + cat_value_tokens
            token_groups.append(cat_tokens)
            mask_groups.append(cat_mask_values)

        tokens = torch.cat(token_groups, dim=1)
        masks = torch.cat(mask_groups, dim=1)
        tokens = tokens * masks.unsqueeze(-1)
        return tokens, masks

    def _validate_cat_range(self, values: Tensor) -> None:
        if torch.any(values < 0):
            raise ValueError("Categorical values must be non-negative class ids.")
        if self.n_cat_features == 0:
            return
        sizes = self.category_sizes_tensor.to(device=values.device).unsqueeze(0)
        if torch.any(values >= sizes.expand_as(values)):
            raise ValueError("Categorical values exceed their feature cardinality.")

    def _validate_role_matrix(
        self,
        values: Tensor | None,
        name: str,
        expected_features: int,
    ) -> Tensor:
        if values is None:
            raise ValueError(
                f"{self.role}.{name} must not be None when its features exist."
            )
        if values.ndim != 2:
            raise ValueError(f"{self.role}.{name} must have shape (batch, features).")
        if values.shape[1] != expected_features:
            raise ValueError(
                f"Expected {self.role}.{name} to have {expected_features} features, "
                f"got {values.shape[1]}."
            )
        return values


class EDDISourceEncoder(nn.Module):
    """Encode tabular tokens with a shared MLP and mask-aware aggregation."""

    def __init__(
        self,
        *,
        role: str,
        n_num_features: int,
        category_sizes: Sequence[int],
        embedding_dim: int,
        hidden_layers: Sequence[int] = (),
        output_dim: int,
        aggregation: str = "sum",
        tokenizer: TabularFeatureTokenizer | None = None,
    ) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        aggregation = aggregation.lower()
        if aggregation not in {"sum", "mean", "max"}:
            raise ValueError("aggregation must be one of: sum, mean, max.")

        self.tokenizer = (
            TabularFeatureTokenizer(
                role=role,
                n_num_features=n_num_features,
                category_sizes=category_sizes,
                embedding_dim=embedding_dim,
            )
            if tokenizer is None
            else tokenizer
        )
        self.role = self.tokenizer.role
        self.n_num_features = self.tokenizer.n_num_features
        self.category_sizes = self.tokenizer.category_sizes
        self.n_cat_features = self.tokenizer.n_cat_features
        self.embedding_dim = self.tokenizer.embedding_dim
        self._output_dim = int(output_dim)
        self.aggregation = aggregation

        layers: list[nn.Module] = []
        in_dim = self.embedding_dim
        for hidden_dim in hidden_layers:
            hidden_dim = int(hidden_dim)
            if hidden_dim <= 0:
                raise ValueError("hidden_layers must contain positive dimensions.")
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self._output_dim))
        self.token_network = nn.Sequential(*layers)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def n_tokens(self) -> int:
        return self.tokenizer.n_tokens

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        self._move_legacy_tokenizer_keys(state_dict, prefix)
        bias_key = prefix + "tokenizer.num_feature_bias"
        if self.tokenizer.num_feature_bias is not None and bias_key not in state_dict:
            state_dict[bias_key] = torch.zeros_like(self.tokenizer.num_feature_bias)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        num_features: Tensor | None,
        cat_features: Tensor | None,
        num_mask: Tensor | None,
        cat_mask: Tensor | None,
    ) -> Tensor:
        encoded_tokens, masks = self.encode_tokens(
            num_features=num_features,
            cat_features=cat_features,
            num_mask=num_mask,
            cat_mask=cat_mask,
        )

        if self.aggregation == "sum":
            return encoded_tokens.sum(dim=1)
        if self.aggregation == "mean":
            denominator = masks.sum(dim=1, keepdim=True).clamp_min(1.0)
            return encoded_tokens.sum(dim=1) / denominator

        masked_tokens = encoded_tokens.masked_fill(masks.unsqueeze(-1) == 0, -torch.inf)
        pooled = masked_tokens.max(dim=1).values
        return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))

    def encode_tokens(
        self,
        *,
        num_features: Tensor | None,
        cat_features: Tensor | None,
        num_mask: Tensor | None,
        cat_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        tokens, masks = self.tokenizer(
            num_features=num_features,
            cat_features=cat_features,
            num_mask=num_mask,
            cat_mask=cat_mask,
        )
        encoded_tokens = self.token_network(tokens)
        encoded_tokens = encoded_tokens * masks.unsqueeze(-1)
        return encoded_tokens, masks

    def _move_legacy_tokenizer_keys(self, state_dict, prefix: str) -> None:
        legacy_names = (
            "num_feature_embedding.weight",
            "num_feature_bias",
            "cat_feature_embedding.weight",
            "cat_value_embedding",
            "category_sizes_tensor",
        )
        for name in legacy_names:
            old_key = prefix + name
            new_key = prefix + "tokenizer." + name
            if old_key in state_dict and new_key not in state_dict:
                state_dict[new_key] = state_dict[old_key]
            state_dict.pop(old_key, None)


class MLPSourceEncoder(nn.Module):
    """Encode flattened source values, one-hot categories, and masks with an MLP."""

    def __init__(
        self,
        *,
        n_num_features: int,
        category_sizes: Sequence[int],
        hidden_layers: Sequence[int] = (),
        output_dim: int,
        dropout: float = 0.0,
        activation: str = "relu",
        normalization: str = "none",
    ) -> None:
        super().__init__()
        if n_num_features < 0:
            raise ValueError("n_num_features must not be negative.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0, 1).")
        self.n_num_features = int(n_num_features)
        self.category_sizes = [int(size) for size in category_sizes]
        if any(size <= 0 for size in self.category_sizes):
            raise ValueError("All category sizes must be positive.")
        self.n_cat_features = len(self.category_sizes)
        self._output_dim = int(output_dim)
        self.input_dim = (
            self.n_num_features
            + sum(max(size - 1, 0) for size in self.category_sizes)
            + self.n_num_features
        )
        if self.input_dim <= 0:
            raise ValueError("At least one source feature is required.")
        self.network = _make_local_mlp(
            input_dim=self.input_dim,
            hidden_layers=hidden_layers,
            output_dim=self._output_dim,
            dropout=dropout,
            activation=activation,
            normalization=normalization,
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        num_features: Tensor | None,
        cat_features: Tensor | None,
        num_mask: Tensor | None,
        cat_mask: Tensor | None,
    ) -> Tensor:
        batch_size = _infer_batch_size(num_features, cat_features, num_mask, cat_mask)
        device = _infer_device(num_features, cat_features, num_mask, cat_mask)
        parts = []

        if self.n_num_features > 0:
            num_values = _validate_matrix(
                num_features,
                name="source.num_features",
                expected_features=self.n_num_features,
            ).to(device=device, dtype=torch.float32)
            num_mask_values = _validate_matrix(
                num_mask,
                name="source.num_mask",
                expected_features=self.n_num_features,
            ).to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
            parts.extend([num_values * num_mask_values, num_mask_values])
        else:
            num_mask_values = torch.empty((batch_size, 0), device=device)

        if self.n_cat_features > 0:
            cat_values = _validate_matrix(
                cat_features,
                name="source.cat_features",
                expected_features=self.n_cat_features,
            ).to(device=device, dtype=torch.long)
            cat_mask_values = _validate_matrix(
                cat_mask,
                name="source.cat_mask",
                expected_features=self.n_cat_features,
            ).to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
            one_hot_parts = []
            for index, size in enumerate(self.category_sizes):
                if size <= 1:
                    continue
                values = cat_values[:, index].clamp(min=0, max=size - 1)
                non_missing = (values > 0).to(dtype=torch.float32).unsqueeze(1)
                shifted = (values - 1).clamp(min=0, max=size - 2)
                one_hot = F.one_hot(shifted, num_classes=size - 1).to(dtype=torch.float32)
                one_hot_parts.append(one_hot * cat_mask_values[:, index : index + 1] * non_missing)
            if one_hot_parts:
                parts.append(torch.cat(one_hot_parts, dim=1))
        else:
            cat_mask_values = torch.empty((batch_size, 0), device=device)

        _ = num_mask_values, cat_mask_values
        return self.network(torch.cat(parts, dim=1))


def make_eddi_source_encoder(
    *,
    role: str,
    n_num_features: int,
    category_sizes: Sequence[int],
    embedding_dim: int,
    hidden_layers: Sequence[int] = (),
    output_dim: int,
    aggregation: str = "sum",
) -> EDDISourceEncoder:
    """Build the EDDI source encoder used by the CVAE."""

    return EDDISourceEncoder(
        role=role,
        n_num_features=n_num_features,
        category_sizes=category_sizes,
        embedding_dim=embedding_dim,
        hidden_layers=hidden_layers,
        output_dim=output_dim,
        aggregation=aggregation,
    )


def make_mlp_source_encoder(
    *,
    n_num_features: int,
    category_sizes: Sequence[int],
    hidden_layers: Sequence[int] = (),
    output_dim: int,
    dropout: float = 0.0,
    activation: str = "relu",
    normalization: str = "none",
) -> MLPSourceEncoder:
    """Build a flattened MLP source encoder for ablation experiments."""

    return MLPSourceEncoder(
        n_num_features=n_num_features,
        category_sizes=category_sizes,
        hidden_layers=hidden_layers,
        output_dim=output_dim,
        dropout=dropout,
        activation=activation,
        normalization=normalization,
    )


def make_tabular_feature_tokenizer(
    *,
    role: str,
    n_num_features: int,
    category_sizes: Sequence[int],
    embedding_dim: int,
) -> TabularFeatureTokenizer:
    """Build the tokenizer-only first stage shared by EDDI and transformers."""

    return TabularFeatureTokenizer(
        role=role,
        n_num_features=n_num_features,
        category_sizes=category_sizes,
        embedding_dim=embedding_dim,
    )


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


def _validate_matrix(values: Tensor | None, *, name: str, expected_features: int) -> Tensor:
    if values is None:
        raise ValueError(f"{name} must not be None when its features exist.")
    if values.ndim != 2:
        raise ValueError(f"{name} must have shape (batch, features).")
    if values.shape[1] != expected_features:
        raise ValueError(f"Expected {name} to have {expected_features} features, got {values.shape[1]}.")
    return values


def _make_local_mlp(
    *,
    input_dim: int,
    hidden_layers: Sequence[int],
    output_dim: int,
    dropout: float,
    activation: str,
    normalization: str,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = int(input_dim)
    normalization = normalization.strip().lower().replace("-", "_")
    if normalization in {"", "none"}:
        normalization = "none"
    if normalization not in {"none", "batch_norm", "layer_norm"}:
        raise ValueError("normalization must be one of: none, batch_norm, layer_norm.")
    for hidden_dim in hidden_layers:
        hidden_dim = int(hidden_dim)
        if hidden_dim <= 0:
            raise ValueError("hidden_layers must contain positive dimensions.")
        layers.append(nn.Linear(current_dim, hidden_dim))
        if normalization == "batch_norm":
            layers.append(nn.BatchNorm1d(hidden_dim))
        elif normalization == "layer_norm":
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(_make_local_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, int(output_dim)))
    return nn.Sequential(*layers)


def _make_local_activation(name: str) -> nn.Module:
    normalized = name.strip().lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name!r}")
