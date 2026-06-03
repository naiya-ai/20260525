"""Plot 2D latent prior/posterior density heatmaps for CVAE test samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from eval.cvae_common import build_model, load_checkpoint_config_and_schema
from train.dataset import create_grouped_cvae_dataloader, load_grouped_cvae_schema


def main() -> None:
    args = parse_args()
    config, schema = load_checkpoint_config_and_schema(args)
    config["data"]["target_group"] = args.target_group
    if args.dataset_root:
        config["data"]["dataset_root"] = args.dataset_root
    if args.dataset_name:
        config["data"]["dataset_name"] = args.dataset_name

    if int(config["model"]["latent_dim"]) != 2:
        raise ValueError("Latent heatmap plotting requires model.latent_dim == 2.")

    schema = load_grouped_cvae_schema(config)
    model = build_model(config, schema)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    loader = create_grouped_cvae_dataloader(
        config,
        split=args.split,
        batch_size=max(args.num_samples * 4, args.num_samples),
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    batch = next(iter(loader))
    keep = complete_target_indices(batch, args.num_samples)
    batch = {key: value[keep] for key, value in batch.items()}

    with torch.no_grad():
        outputs = model(
            source_num=batch["source_num"],
            source_cat=batch["source_cat"],
            source_num_mask=batch["source_num_mask"],
            source_cat_mask=batch["source_cat_mask"],
            target_num=batch["target_num"],
            target_cat=batch["target_cat"],
            target_num_mask=batch["target_num_mask"],
            target_cat_mask=batch["target_cat_mask"],
            sample_latent=False,
        )

    posterior_mu = outputs["mu"].cpu().numpy()
    posterior_logvar = outputs["logvar"].cpu().numpy()
    prior_mu = outputs["prior_mu"].cpu().numpy()
    prior_logvar = outputs["prior_logvar"].cpu().numpy()
    target_num = batch["target_num"].cpu().numpy()

    plot_heatmaps(
        output=args.output,
        prior_mu=prior_mu,
        prior_logvar=prior_logvar,
        posterior_mu=posterior_mu,
        posterior_logvar=posterior_logvar,
        target_num=target_num,
        title=args.title,
    )
    print(args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-group", default="diabetes")
    parser.add_argument("--dataset-root")
    parser.add_argument("--dataset-name")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/plots/diabetes_d512_z2_latent_prior_posterior_heatmaps.png"),
    )
    parser.add_argument(
        "--title",
        default="Latent prior/posterior distributions on test samples",
    )
    return parser.parse_args()


def complete_target_indices(batch: dict[str, torch.Tensor], count: int) -> torch.Tensor:
    target_num_mask = batch["target_num_mask"].bool()
    target_cat_mask = batch["target_cat_mask"].bool()
    complete_num = target_num_mask.all(dim=1)
    complete_cat = (
        target_cat_mask.all(dim=1)
        if target_cat_mask.shape[1] > 0
        else torch.ones_like(complete_num)
    )
    keep = torch.nonzero(complete_num & complete_cat, as_tuple=False).flatten()
    if len(keep) < count:
        raise ValueError(f"Only found {len(keep)} complete target samples, need {count}.")
    return keep[:count]


def plot_heatmaps(
    *,
    output: Path,
    prior_mu: np.ndarray,
    prior_logvar: np.ndarray,
    posterior_mu: np.ndarray,
    posterior_logvar: np.ndarray,
    target_num: np.ndarray,
    title: str,
) -> None:
    n_samples = prior_mu.shape[0]
    fig, axes = plt.subplots(n_samples, 2, figsize=(8, 3.2 * n_samples), squeeze=False)
    for sample_idx in range(n_samples):
        mu_pair = np.stack([prior_mu[sample_idx], posterior_mu[sample_idx]], axis=0)
        std_pair = np.exp(0.5 * np.stack([prior_logvar[sample_idx], posterior_logvar[sample_idx]], axis=0))
        low = np.min(mu_pair - 4.0 * std_pair, axis=0)
        high = np.max(mu_pair + 4.0 * std_pair, axis=0)
        pad = np.maximum((high - low) * 0.05, 0.25)
        x = np.linspace(low[0] - pad[0], high[0] + pad[0], 180)
        y = np.linspace(low[1] - pad[1], high[1] + pad[1], 180)
        xx, yy = np.meshgrid(x, y)

        panels = (
            ("prior p(z|x)", prior_mu[sample_idx], prior_logvar[sample_idx], "Blues"),
            ("posterior q(z|x,y)", posterior_mu[sample_idx], posterior_logvar[sample_idx], "Reds"),
        )
        for ax, (title, mu, logvar, cmap) in zip(axes[sample_idx], panels):
            density = diagonal_gaussian_pdf(xx, yy, mu, logvar)
            ax.imshow(
                density,
                extent=[x.min(), x.max(), y.min(), y.max()],
                origin="lower",
                aspect="auto",
                cmap=cmap,
            )
            ax.scatter([mu[0]], [mu[1]], c="black", s=18, marker="x", linewidths=1.5)
            ax.set_title(
                f"sample {sample_idx} {title}\n"
                f"mu=({mu[0]:.2f},{mu[1]:.2f}) std=({np.exp(0.5 * logvar[0]):.2f},{np.exp(0.5 * logvar[1]):.2f})"
            )
            ax.set_xlabel("z1")
            ax.set_ylabel("z2")
            ax.grid(alpha=0.15)
        axes[sample_idx, 0].text(
            0.02,
            0.96,
            f"target z-space num: glu={target_num[sample_idx, 0]:.2f}, HbA1c={target_num[sample_idx, 1]:.2f}",
            transform=axes[sample_idx, 0].transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def diagonal_gaussian_pdf(
    xx: np.ndarray,
    yy: np.ndarray,
    mu: np.ndarray,
    logvar: np.ndarray,
) -> np.ndarray:
    var = np.exp(logvar)
    exponent = -0.5 * (((xx - mu[0]) ** 2 / var[0]) + ((yy - mu[1]) ** 2 / var[1]))
    norm = 1.0 / (2.0 * np.pi * np.sqrt(var[0] * var[1]))
    return norm * np.exp(exponent)


if __name__ == "__main__":
    main()
