"""Disease classifier heads on top of a masked source autoencoder backbone."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.conditional_vae import _make_mlp
from models.masked_autoencoder import MaskedSourceAutoencoder


class MaskedAEClassifier(nn.Module):
    """Predict a binary disease label from a pretrained masked-AE source encoder."""

    def __init__(
        self,
        *,
        autoencoder: MaskedSourceAutoencoder,
        head_hidden_layers: Sequence[int] = (),
        dropout: float = 0.0,
        activation: str = "relu",
        batch_norm: bool = False,
        freeze_backbone: bool = False,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0, 1).")
        if pos_weight is not None and pos_weight <= 0:
            raise ValueError("pos_weight must be positive when set.")
        self.source_encoder = autoencoder.source_encoder
        self.condition_dim = int(autoencoder.condition_dim)
        self.freeze_backbone = bool(freeze_backbone)
        self.head = _make_mlp(
            input_dim=self.condition_dim,
            hidden_layers=head_hidden_layers,
            output_dim=1,
            dropout=dropout,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.register_buffer(
            "pos_weight",
            None if pos_weight is None else torch.tensor(float(pos_weight)),
        )
        if self.freeze_backbone:
            self._set_backbone_requires_grad(False)

    def train(self, mode: bool = True) -> "MaskedAEClassifier":
        super().train(mode)
        if self.freeze_backbone:
            self.source_encoder.eval()
        return self

    def encode_source(
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

    def forward(
        self,
        *,
        source_num: Tensor | None,
        source_cat: Tensor | None,
        source_num_mask: Tensor | None,
        source_cat_mask: Tensor | None,
    ) -> dict[str, Tensor]:
        embedding = self.encode_source(
            source_num=source_num,
            source_cat=source_cat,
            source_num_mask=source_num_mask,
            source_cat_mask=source_cat_mask,
        )
        logit = self.head(embedding).squeeze(-1)
        return {
            "embedding": embedding,
            "logit": logit,
            "probability": torch.sigmoid(logit),
        }

    def loss(
        self,
        *,
        source_num: Tensor | None,
        source_cat: Tensor | None,
        source_num_mask: Tensor | None,
        source_cat_mask: Tensor | None,
        target: Tensor,
    ) -> dict[str, Tensor]:
        outputs = self(
            source_num=source_num,
            source_cat=source_cat,
            source_num_mask=source_num_mask,
            source_cat_mask=source_cat_mask,
        )
        target = target.to(device=outputs["logit"].device, dtype=outputs["logit"].dtype)
        pos_weight = (
            None
            if self.pos_weight is None
            else self.pos_weight.to(device=outputs["logit"].device, dtype=outputs["logit"].dtype)
        )
        loss = F.binary_cross_entropy_with_logits(
            outputs["logit"],
            target,
            pos_weight=pos_weight,
        )
        return {"loss": loss, "target": target, **outputs}

    def _set_backbone_requires_grad(self, requires_grad: bool) -> None:
        for parameter in self.source_encoder.parameters():
            parameter.requires_grad_(requires_grad)
