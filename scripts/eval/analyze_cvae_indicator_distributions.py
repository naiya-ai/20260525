"""Analyze numerical indicator distributions from a trained CVAE."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from eval.cvae_common import (
    build_model,
    decode_source_context,
    disease_predictions,
    inverse_gaussian_quantile,
    load_checkpoint_config_and_schema,
    read_target_metadata,
    resolve_device,
    set_seed,
)
from train.dataset import create_grouped_cvae_dataloader, load_grouped_cvae_schema


NUMERICAL_DISEASES = (
    "diabetes",
    "hypertension",
    "dyslipidemia",
    "liver_disease",
    "kidney_disease",
    "anemia",
)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = args.output_dir
    scatter_dir = output_dir / "scatter"
    quantile_dir = output_dir / "quantile_histograms"
    sample_dir = output_dir / "top10_random_sample_histograms"
    label_dir = output_dir / "label_stratified_model_distributions"
    data_dir = output_dir / "data"
    directories = [scatter_dir, quantile_dir, sample_dir, data_dir]
    if args.label_distribution_num_samples > 0:
        directories.append(label_dir)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for disease in args.diseases:
        checkpoint = args.checkpoint_root / disease / args.checkpoint_name
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        disease_rows, disease_selected = analyze_disease(
            checkpoint=checkpoint,
            disease=disease,
            dataset_name=args.dataset_name,
            dataset_root=args.dataset_root,
            device=resolve_device(args.device),
            batch_size=args.batch_size,
            quantile_num_samples=args.quantile_num_samples,
            single_num_samples=args.single_num_samples,
            seed=args.seed,
            value_scale=args.value_scale,
            scatter_dir=scatter_dir,
            quantile_dir=quantile_dir,
            sample_dir=sample_dir,
            label_dir=label_dir,
            data_dir=data_dir,
            label_distribution_num_samples=args.label_distribution_num_samples,
        )
        rows.extend(disease_rows)
        selected_rows.extend(disease_selected)

    pd.DataFrame(rows).to_csv(output_dir / "indicator_summary.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(output_dir / "top10_random_selected_samples.csv", index=False)
    print(f"wrote {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="checkpoint_best.pt")
    parser.add_argument("--dataset-root", default="datasets/preprocessed/gaussian_quantile")
    parser.add_argument("--dataset-name", default="harmonized_knhanes_1998_2024")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--quantile-num-samples", type=int, default=100)
    parser.add_argument("--single-num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diseases", nargs="+", default=list(NUMERICAL_DISEASES))
    parser.add_argument("--value-scale", choices=("original", "transformed"), default="original")
    parser.add_argument(
        "--label-distribution-num-samples",
        type=int,
        default=0,
        help="If positive, also plot data/model/model-by-disease-label distributions with this many prior samples per row.",
    )
    return parser.parse_args()


def analyze_disease(
    *,
    checkpoint: Path,
    disease: str,
    dataset_name: str,
    dataset_root: str,
    device: torch.device,
    batch_size: int,
    quantile_num_samples: int,
    single_num_samples: int,
    seed: int,
    value_scale: str,
    scatter_dir: Path,
    quantile_dir: Path,
    sample_dir: Path,
    label_dir: Path,
    data_dir: Path,
    label_distribution_num_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    args = argparse.Namespace(checkpoint=checkpoint)
    config, _ = load_checkpoint_config_and_schema(args)
    config["data"]["target_group"] = disease
    config["data"]["dataset_name"] = dataset_name
    config["data"]["dataset_root"] = dataset_root

    schema = load_grouped_cvae_schema(config)
    model = build_model(config, schema)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(device).eval()
    target_metadata = read_target_metadata(config)
    features = list(target_metadata["features"]["num"])
    if not features:
        return [], []

    loader = create_grouped_cvae_dataloader(
        config,
        split="test",
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        seed=seed,
    )
    arrays = collect_test_arrays(
        model=model,
        loader=loader,
        target_metadata=target_metadata,
        device=device,
        value_scale=value_scale,
    )
    if label_distribution_num_samples > 0:
        if value_scale != "original":
            raise ValueError("--label-distribution-num-samples requires --value-scale original.")
        plot_label_stratified_distributions(
            output_dir=label_dir,
            data_dir=data_dir,
            disease=disease,
            features=features,
            model=model,
            arrays=arrays,
            schema=schema,
            config=config,
            target_metadata=target_metadata,
            device=device,
            num_samples=label_distribution_num_samples,
        )
    write_scatter_data(
        data_dir / f"{disease}_prior_sample_scatter.csv",
        disease,
        features,
        arrays,
        generated_key="prior_sample_values",
        generated_fieldname="prior_sample_value",
    )
    plot_scatter(
        scatter_dir / f"{disease}_prior_sample_scatter.png",
        disease,
        features,
        arrays,
        generated_key="prior_sample_values",
        generated_label="decoded prior sample",
        color="#059669",
        title_suffix="decoded conditional prior sample",
        value_scale=value_scale,
    )

    rng = np.random.default_rng(seed + stable_int(disease))
    summary_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for feature_idx, feature in enumerate(features):
        true_values = arrays["true_values"][:, feature_idx]
        observed = arrays["target_num_mask"][:, feature_idx] > 0
        observed_indices = np.flatnonzero(observed & np.isfinite(true_values))
        if observed_indices.size == 0:
            continue

        sorted_indices = observed_indices[np.argsort(true_values[observed_indices])[::-1]]
        quantile_groups = true_value_decile_bins(sorted_indices)
        generated_bins: list[tuple[str, str, np.ndarray]] = []
        for bin_key, bin_title, bin_indices in quantile_groups:
            generated_values = sample_feature_values(
                model=model,
                arrays=arrays,
                target_metadata=target_metadata,
                feature_idx=feature_idx,
                indices=bin_indices,
                num_samples=quantile_num_samples,
                device=device,
                value_scale=value_scale,
            )
            generated_bins.append((bin_key, bin_title, generated_values))
        plot_quantile_histogram(
            quantile_dir / f"{disease}_{feature}_quantile_hist.png",
            disease=disease,
            feature=feature,
            value_scale=value_scale,
            reference=true_values[observed_indices],
            generated_bins=generated_bins,
        )
        write_distribution_values(
            data_dir / f"{disease}_{feature}_quantile_generated_values.csv",
            disease=disease,
            feature=feature,
            generated_bins=generated_bins,
        )

        top10 = quantile_groups[0][2]
        top10_20 = quantile_groups[1][2]
        generated_top10 = generated_bins[0][2]
        generated_top10_20 = generated_bins[1][2]
        chosen = rng.choice(top10, size=min(4, len(top10)), replace=False)
        single_values: list[np.ndarray] = []
        for idx in chosen:
            values = sample_feature_values(
                model=model,
                arrays=arrays,
                target_metadata=target_metadata,
                feature_idx=feature_idx,
                indices=np.asarray([idx], dtype=np.int64),
                num_samples=single_num_samples,
                device=device,
                value_scale=value_scale,
            )
            single_values.append(values)
            selected_rows.append(
                {
                    "disease": disease,
                    "feature": feature,
                    "test_position": int(idx),
                    "true_value": float(true_values[idx]),
                    "num_samples": int(single_num_samples),
                }
            )
        plot_single_sample_histograms(
            sample_dir / f"{disease}_{feature}_top10_random4_hist.png",
            disease=disease,
            feature=feature,
            value_scale=value_scale,
            reference=true_values[observed_indices],
            selected_indices=chosen,
            true_values=true_values,
            generated_values=single_values,
        )
        write_single_sample_values(
            data_dir / f"{disease}_{feature}_top10_random4_generated_values.csv",
            disease=disease,
            feature=feature,
            selected_indices=chosen,
            true_values=true_values,
            generated_values=single_values,
        )
        summary_row = {
            "disease": disease,
            "feature": feature,
            "n_observed_test": int(observed_indices.size),
            "top10_n": int(len(top10)),
            "top10_20_n": int(len(top10_20)),
            "prior_sample_pearson": safe_corr(
                true_values[observed_indices],
                arrays["prior_sample_values"][observed_indices, feature_idx],
            ),
            "true_mean": float(np.nanmean(true_values[observed_indices])),
            "prior_sample_mean": float(np.nanmean(arrays["prior_sample_values"][observed_indices, feature_idx])),
            "top10_generated_mean": float(np.nanmean(generated_top10)),
            "top10_20_generated_mean": float(np.nanmean(generated_top10_20)),
            "value_scale": value_scale,
        }
        for bin_key, _, values in generated_bins:
            summary_row[f"{bin_key}_generated_mean"] = float(np.nanmean(values))
        summary_rows.append(summary_row)
    return summary_rows, selected_rows


@torch.no_grad()
def collect_test_arrays(
    *,
    model,
    loader,
    target_metadata: dict[str, Any],
    device: torch.device,
    value_scale: str,
) -> dict[str, np.ndarray | torch.Tensor]:
    source_num_parts = []
    source_cat_parts = []
    source_num_mask_parts = []
    source_cat_mask_parts = []
    target_num_parts = []
    target_num_mask_parts = []
    prior_sample_parts = []
    for batch in loader:
        cpu_batch = {key: value.detach().cpu() for key, value in batch.items()}
        source_num_parts.append(cpu_batch["source_num"])
        source_cat_parts.append(cpu_batch["source_cat"])
        source_num_mask_parts.append(cpu_batch["source_num_mask"])
        source_cat_mask_parts.append(cpu_batch["source_cat_mask"])
        target_num_parts.append(cpu_batch["target_num"])
        target_num_mask_parts.append(cpu_batch["target_num_mask"])

        device_batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        condition = model.encode_condition(
            source_num=device_batch["source_num"],
            source_cat=device_batch["source_cat"],
            source_num_mask=device_batch["source_num_mask"],
            source_cat_mask=device_batch["source_cat_mask"],
        )
        z_sample = model.sample_prior(condition, sample=True)
        sampled_decoded = decode_model(
            model=model,
            z=z_sample,
            condition=condition,
            batch=device_batch,
        )
        prior_sample_parts.append(sampled_decoded["num_mean"].detach().cpu())

    source_num = torch.cat(source_num_parts, dim=0)
    source_cat = torch.cat(source_cat_parts, dim=0)
    source_num_mask = torch.cat(source_num_mask_parts, dim=0)
    source_cat_mask = torch.cat(source_cat_mask_parts, dim=0)
    target_num = torch.cat(target_num_parts, dim=0)
    target_num_mask = torch.cat(target_num_mask_parts, dim=0)
    prior_sample_num = torch.cat(prior_sample_parts, dim=0)
    target_num_values = target_num.numpy().astype(np.float64)
    prior_sample_values = prior_sample_num.numpy().astype(np.float64)
    if value_scale == "original":
        target_num_values = inverse_num_matrix(target_num_values, target_metadata)
        prior_sample_values = inverse_num_matrix(prior_sample_values, target_metadata)
    elif value_scale != "transformed":
        raise ValueError(f"Unsupported value scale: {value_scale}")

    return {
        "source_num": source_num,
        "source_cat": source_cat,
        "source_num_mask": source_num_mask,
        "source_cat_mask": source_cat_mask,
        "target_num": target_num,
        "target_num_mask": target_num_mask.numpy(),
        "true_values": target_num_values,
        "prior_sample_values": prior_sample_values,
    }


def inverse_num_matrix(values: np.ndarray, target_metadata: dict[str, Any]) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    for idx, state in enumerate(target_metadata["gaussian_quantile"]["features"]):
        out[:, idx] = inverse_gaussian_quantile(values[:, idx], state)
    return out


@torch.no_grad()
def sample_feature_values(
    *,
    model,
    arrays: dict[str, Any],
    target_metadata: dict[str, Any],
    feature_idx: int,
    indices: np.ndarray,
    num_samples: int,
    device: torch.device,
    value_scale: str,
) -> np.ndarray:
    if len(indices) == 0:
        return np.asarray([], dtype=np.float64)
    idx = torch.as_tensor(indices, dtype=torch.long)
    source_num = arrays["source_num"].index_select(0, idx).to(device)
    source_cat = arrays["source_cat"].index_select(0, idx).to(device)
    source_num_mask = arrays["source_num_mask"].index_select(0, idx).to(device)
    source_cat_mask = arrays["source_cat_mask"].index_select(0, idx).to(device)
    condition = model.encode_condition(
        source_num=source_num,
        source_cat=source_cat,
        source_num_mask=source_num_mask,
        source_cat_mask=source_cat_mask,
    )
    values = []
    for _ in range(num_samples):
        z = model.sample_prior(condition, sample=True)
        decoded = decode_model(
            model=model,
            z=z,
            condition=condition,
            batch={
                "source_num": source_num,
                "source_cat": source_cat,
                "source_num_mask": source_num_mask,
                "source_cat_mask": source_cat_mask,
            },
        )
        transformed = decoded["num_mean"][:, feature_idx].detach().cpu().numpy()
        if value_scale == "original":
            state = target_metadata["gaussian_quantile"]["features"][feature_idx]
            values.append(inverse_gaussian_quantile(transformed, state))
        elif value_scale == "transformed":
            values.append(transformed.astype(np.float64))
        else:
            raise ValueError(f"Unsupported value scale: {value_scale}")
    return np.concatenate(values).astype(np.float64)


def decode_model(*, model, z: torch.Tensor, condition: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if hasattr(model, "decode_from_source"):
        return model.decode_from_source(
            z,
            source_num=batch["source_num"],
            source_cat=batch["source_cat"],
            source_num_mask=batch["source_num_mask"],
            source_cat_mask=batch["source_cat_mask"],
            condition=condition,
        )
    return model.decode(z, condition)


def true_value_decile_bins(sorted_indices: np.ndarray) -> list[tuple[str, str, np.ndarray]]:
    n = len(sorted_indices)
    bins: list[tuple[str, str, np.ndarray]] = []
    for bin_idx in range(10):
        low = bin_idx * 10
        high = (bin_idx + 1) * 10
        start = int(np.floor(n * bin_idx / 10.0))
        end = n if bin_idx == 9 else int(np.floor(n * (bin_idx + 1) / 10.0))
        if end <= start and start < n:
            end = start + 1
        key = f"true_value_{low:02d}_{high:02d}"
        title = f"Generated from true-value {low}-{high}% rows"
        bins.append((key, title, sorted_indices[start:end]))
    return bins


def plot_scatter(
    path: Path,
    disease: str,
    features: list[str],
    arrays: dict[str, Any],
    *,
    generated_key: str,
    generated_label: str,
    color: str,
    title_suffix: str,
    value_scale: str,
) -> None:
    cols = min(3, len(features))
    rows = int(np.ceil(len(features) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes_flat = axes.ravel()
    for ax, feature_idx, feature in zip(axes_flat, range(len(features)), features):
        true_values = arrays["true_values"][:, feature_idx]
        generated = arrays[generated_key][:, feature_idx]
        mask = (arrays["target_num_mask"][:, feature_idx] > 0) & np.isfinite(true_values) & np.isfinite(generated)
        ax.scatter(true_values[mask], generated[mask], s=7, alpha=0.35, color=color)
        if np.any(mask):
            lo = float(np.nanmin([true_values[mask].min(), generated[mask].min()]))
            hi = float(np.nanmax([true_values[mask].max(), generated[mask].max()]))
            ax.plot([lo, hi], [lo, hi], color="#111827", linewidth=0.8, linestyle="--")
        ax.set_title(feature)
        ax.set_xlabel(value_axis_label("true value", value_scale))
        ax.set_ylabel(value_axis_label(generated_label, value_scale))
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
    for ax in axes_flat[len(features):]:
        ax.axis("off")
    fig.suptitle(f"{disease}: true indicator vs {title_suffix}", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_quantile_histogram(
    path: Path,
    *,
    disease: str,
    feature: str,
    value_scale: str,
    reference: np.ndarray,
    generated_bins: list[tuple[str, str, np.ndarray]],
) -> None:
    fig, axes = plt.subplots(1 + len(generated_bins), 1, figsize=(8, 2.05 * (1 + len(generated_bins))), sharex=True)
    items = [
        ("Observed full test reference", reference, "#374151"),
    ]
    palette = [
        "#dc2626",
        "#f97316",
        "#f59e0b",
        "#84cc16",
        "#22c55e",
        "#14b8a6",
        "#06b6d4",
        "#3b82f6",
        "#8b5cf6",
        "#ec4899",
    ]
    items.extend(
        (title, values, palette[idx % len(palette)])
        for idx, (_, title, values) in enumerate(generated_bins)
    )
    bins = shared_bins([reference, *[values for _, _, values in generated_bins]])
    for ax, (title, values, color) in zip(axes, items):
        ax.hist(values[np.isfinite(values)], bins=bins, color=color, alpha=0.8, density=True)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("density")
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
    axes[-1].set_xlabel(value_axis_label(feature, value_scale))
    fig.suptitle(
        f"{disease} / {feature}: reference and true-value decile generated distributions",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_single_sample_histograms(
    path: Path,
    *,
    disease: str,
    feature: str,
    value_scale: str,
    reference: np.ndarray,
    selected_indices: np.ndarray,
    true_values: np.ndarray,
    generated_values: list[np.ndarray],
) -> None:
    n = len(generated_values)
    fig, axes = plt.subplots(n + 1, 1, figsize=(8, 2.6 * (n + 1)), sharex=True)
    all_values = [reference, *generated_values]
    bins = shared_bins(all_values)
    axes[0].hist(reference[np.isfinite(reference)], bins=bins, color="#374151", alpha=0.8, density=True)
    axes[0].set_title("Observed full test reference")
    axes[0].grid(True, color="#e5e7eb", linewidth=0.8)
    for ax, idx, values in zip(axes[1:], selected_indices, generated_values):
        ax.hist(values[np.isfinite(values)], bins=bins, color="#2563eb", alpha=0.8, density=True)
        ax.axvline(true_values[idx], color="#dc2626", linewidth=1.5, label=f"true={true_values[idx]:.3g}")
        ax.set_title(f"Top-10% random sample test_position={int(idx)}")
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.legend(frameon=False)
    for ax in axes:
        ax.set_ylabel("density")
    axes[-1].set_xlabel(value_axis_label(feature, value_scale))
    fig.suptitle(f"{disease} / {feature}: N={len(generated_values[0]) if generated_values else 0} prior samples per selected row", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_label_stratified_distributions(
    *,
    output_dir: Path,
    data_dir: Path,
    disease: str,
    features: list[str],
    model,
    arrays: dict[str, Any],
    schema: dict[str, Any],
    config: dict[str, Any],
    target_metadata: dict[str, Any],
    device: torch.device,
    num_samples: int,
) -> None:
    batch = {
        "source_num": arrays["source_num"],
        "source_cat": arrays["source_cat"],
        "source_num_mask": arrays["source_num_mask"],
        "source_cat_mask": arrays["source_cat_mask"],
    }
    source = decode_source_context(batch=batch, schema=schema, config=config)
    true_indicators = {
        feature: arrays["true_values"][:, feature_idx]
        for feature_idx, feature in enumerate(features)
    }
    labels = np.asarray(disease_predictions(disease, true_indicators, source), dtype=object)

    selected_data: list[dict[str, Any]] = []
    for feature_idx, feature in enumerate(features):
        true_values = arrays["true_values"][:, feature_idx]
        observed = (arrays["target_num_mask"][:, feature_idx] > 0) & np.isfinite(true_values)
        observed_indices = np.flatnonzero(observed)
        false_indices = np.asarray(
            [idx for idx in observed_indices if labels[idx] is False],
            dtype=np.int64,
        )
        true_indices = np.asarray(
            [idx for idx in observed_indices if labels[idx] is True],
            dtype=np.int64,
        )

        model_all = sample_feature_values(
            model=model,
            arrays=arrays,
            target_metadata=target_metadata,
            feature_idx=feature_idx,
            indices=observed_indices,
            num_samples=num_samples,
            device=device,
            value_scale="original",
        )
        model_false = sample_feature_values(
            model=model,
            arrays=arrays,
            target_metadata=target_metadata,
            feature_idx=feature_idx,
            indices=false_indices,
            num_samples=num_samples,
            device=device,
            value_scale="original",
        )
        model_true = sample_feature_values(
            model=model,
            arrays=arrays,
            target_metadata=target_metadata,
            feature_idx=feature_idx,
            indices=true_indices,
            num_samples=num_samples,
            device=device,
            value_scale="original",
        )
        reference = true_values[observed_indices]
        indicator_positive_probs = {
            "data": indicator_positive_probability(
                disease=disease,
                feature=feature,
                values=reference,
                source=source,
                indices=observed_indices,
                num_samples=1,
            ),
            "model_all": indicator_positive_probability(
                disease=disease,
                feature=feature,
                values=model_all,
                source=source,
                indices=observed_indices,
                num_samples=num_samples,
            ),
            "model_false": indicator_positive_probability(
                disease=disease,
                feature=feature,
                values=model_false,
                source=source,
                indices=false_indices,
                num_samples=num_samples,
            ),
            "model_true": indicator_positive_probability(
                disease=disease,
                feature=feature,
                values=model_true,
                source=source,
                indices=true_indices,
                num_samples=num_samples,
            ),
        }
        plot_four_panel_distribution(
            output_dir / f"{disease}_{feature}_label_stratified_model_distribution.png",
            disease=disease,
            feature=feature,
            reference=reference,
            model_all=model_all,
            model_false=model_false,
            model_true=model_true,
            source=source,
            observed_indices=observed_indices,
            n_observed=len(observed_indices),
            n_false=len(false_indices),
            n_true=len(true_indices),
            num_samples=num_samples,
            indicator_positive_probs=indicator_positive_probs,
        )
        write_label_distribution_values(
            data_dir / f"{disease}_{feature}_label_stratified_model_distribution.csv",
            disease=disease,
            feature=feature,
            reference=reference,
            model_all=model_all,
            model_false=model_false,
            model_true=model_true,
        )
        selected_data.append(
            {
                "disease": disease,
                "feature": feature,
                "n_observed": int(len(observed_indices)),
                "n_label_false": int(len(false_indices)),
                "n_label_true": int(len(true_indices)),
                "num_samples_per_row": int(num_samples),
                "data_indicator_positive_probability": indicator_positive_probs["data"],
                "model_all_indicator_positive_probability": indicator_positive_probs["model_all"],
                "model_false_indicator_positive_probability": indicator_positive_probs["model_false"],
                "model_true_indicator_positive_probability": indicator_positive_probs["model_true"],
            }
        )

    pd.DataFrame(selected_data).to_csv(data_dir / f"{disease}_label_stratified_distribution_summary.csv", index=False)


def plot_four_panel_distribution(
    path: Path,
    *,
    disease: str,
    feature: str,
    reference: np.ndarray,
    model_all: np.ndarray,
    model_false: np.ndarray,
    model_true: np.ndarray,
    source: dict[str, np.ndarray],
    observed_indices: np.ndarray,
    n_observed: int,
    n_false: int,
    n_true: int,
    num_samples: int,
    indicator_positive_probs: dict[str, float],
) -> None:
    items = [
        ("data", f"Data distribution (test observed, n={n_observed})", reference),
        ("model_all", f"Model distribution (prior samples, N={num_samples}/row)", model_all),
        ("model_false", f"Model distribution | disease_label = false (n={n_false})", model_false),
        ("model_true", f"Model distribution | disease_label = true (n={n_true})", model_true),
    ]
    bins = shared_bins([values for _, _, values in items])
    indication = disease_indicator_indication(disease, feature, source, observed_indices)
    fig, axes = plt.subplots(4, 1, figsize=(8.4, 8.8), sharex=True)
    for panel_idx, (ax, (key, title, values)) in enumerate(zip(axes, items)):
        finite = values[np.isfinite(values)]
        if finite.size:
            ax.hist(
                finite,
                bins=bins,
                density=True,
                alpha=0.82,
                color="#9CA3AF",
                edgecolor="#6B7280",
                linewidth=0.25,
                zorder=2,
            )
        else:
            ax.text(0.5, 0.5, "No finite values", transform=ax.transAxes, ha="center", va="center", color="#6B7280")
        draw_indication_region(ax, indication, float(bins[0]), float(bins[-1]))
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("density")
        ax.grid(True, color="#E5E7EB", linewidth=0.8)
        if panel_idx == 0:
            ax.text(
                0.01,
                0.88,
                indication["label"],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=12,
                color="#7C2D12",
                bbox={"facecolor": "white", "edgecolor": "#FED7AA", "alpha": 0.9, "pad": 3},
                zorder=30,
            )
        prob = indicator_positive_probs[key]
        prob_text = "P(indicator in disease range): n/a"
        if np.isfinite(prob):
            prob_text = f"P(indicator in disease range): {prob:.3f}"
        ax.text(
            0.01,
            0.74 if panel_idx == 0 else 0.88,
            prob_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            color="#991B1B",
            bbox={"facecolor": "white", "edgecolor": "#FECACA", "alpha": 0.9, "pad": 3},
            zorder=30,
        )
    axes[-1].set_xlabel(feature)
    fig.suptitle(f"{disease} / {feature}: real-value distribution by disease label", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def disease_indicator_indication(
    disease: str,
    feature: str,
    source: dict[str, np.ndarray],
    observed_indices: np.ndarray,
) -> dict[str, Any]:
    if disease == "diabetes":
        thresholds = {"HE_glu": 126.0, "HE_HbA1c": 6.5}
        threshold = thresholds[feature]
        return {"label": f"disease indication: {feature} >= {threshold:g}", "spans": [(threshold, None)], "lines": [threshold]}
    if disease == "hypertension":
        thresholds = {"HE_sbp": 140.0, "HE_dbp": 90.0}
        threshold = thresholds[feature]
        return {"label": f"disease indication: {feature} >= {threshold:g}", "spans": [(threshold, None)], "lines": [threshold]}
    if disease == "dyslipidemia":
        if feature == "HE_chol":
            return {"label": "disease indication: HE_chol >= 240", "spans": [(240.0, None)], "lines": [240.0]}
        if feature == "HE_TG":
            return {"label": "disease indication: HE_TG >= 200", "spans": [(200.0, None)], "lines": [200.0]}
        return {
            "label": "disease indication: HE_HDL_st2 < 40 male (blue) / < 50 female (red)",
            "spans": [(None, 50.0)],
            "lines": [
                {"value": 40.0, "color": "#2563EB"},
                {"value": 50.0, "color": "#DC2626"},
            ],
        }
    if disease == "liver_disease":
        threshold = 40.0
        return {"label": f"disease indication: {feature} >= {threshold:g}", "spans": [(threshold, None)], "lines": [threshold]}
    if disease == "kidney_disease":
        thresholds = kidney_creatinine_thresholds(source, observed_indices)
        finite = thresholds[np.isfinite(thresholds)]
        if finite.size:
            q10, q50, q90 = np.percentile(finite, [10, 50, 90])
            return {
                "label": f"disease indication: eGFR < 60; creatinine cutoff p10/p50/p90={q10:.2f}/{q50:.2f}/{q90:.2f}",
                "spans": [(float(q10), None)],
                "lines": [float(q10), float(q50), float(q90)],
            }
        return {"label": "disease indication: eGFR < 60; cutoff depends on age/sex", "spans": [], "lines": []}
    if disease == "anemia":
        return {
            "label": "disease indication: HE_HB < 13 male (blue) / < 12 female (red)",
            "spans": [(None, 13.0)],
            "lines": [
                {"value": 13.0, "color": "#2563EB"},
                {"value": 12.0, "color": "#DC2626"},
            ],
        }
    return {"label": "disease indication: n/a", "spans": [], "lines": []}


def draw_indication_region(ax, indication: dict[str, Any], x_min: float, x_max: float) -> None:
    for line in indication["lines"]:
        if isinstance(line, dict):
            value = float(line["value"])
            color = str(line.get("color", "#DC2626"))
        else:
            value = float(line)
            color = "#DC2626"
        if x_min <= value <= x_max:
            ax.axvline(value, color=color, linewidth=2.8, linestyle="-", zorder=10)


def indicator_positive_probability(
    *,
    disease: str,
    feature: str,
    values: np.ndarray,
    source: dict[str, np.ndarray],
    indices: np.ndarray,
    num_samples: int,
) -> float:
    finite = np.isfinite(values)
    if values.size == 0 or not finite.any():
        return float("nan")
    if disease == "diabetes":
        thresholds = {"HE_glu": 126.0, "HE_HbA1c": 6.5}
        mask = values >= thresholds[feature]
    elif disease == "hypertension":
        thresholds = {"HE_sbp": 140.0, "HE_dbp": 90.0}
        mask = values >= thresholds[feature]
    elif disease == "dyslipidemia":
        if feature == "HE_chol":
            mask = values >= 240.0
        elif feature == "HE_TG":
            mask = values >= 200.0
        else:
            sex = np.tile(source["sex"][indices], num_samples)
            mask = ((sex == 1) & (values < 40.0)) | ((sex == 2) & (values < 50.0))
            finite = finite & np.isfinite(sex)
    elif disease == "liver_disease":
        mask = values >= 40.0
    elif disease == "kidney_disease":
        age = np.tile(source["age"][indices], num_samples)
        sex = np.tile(source["sex"][indices], num_samples)
        egfr = egfr_from_creatinine(values, age, sex)
        mask = (age > 18) & (egfr < 60)
        finite = finite & np.isfinite(age) & np.isfinite(sex) & np.isfinite(egfr)
    elif disease == "anemia":
        sex = np.tile(source["sex"][indices], num_samples)
        mask = ((sex == 1) & (values < 13.0)) | ((sex == 2) & (values < 12.0))
        finite = finite & np.isfinite(sex)
    else:
        return float("nan")
    if not finite.any():
        return float("nan")
    return float(np.mean(mask[finite]))


def egfr_from_creatinine(crea: np.ndarray, age: np.ndarray, sex: np.ndarray) -> np.ndarray:
    crea = np.asarray(crea, dtype=np.float64)
    age = np.asarray(age, dtype=np.float64)
    sex = np.asarray(sex, dtype=np.float64)
    female_egfr = (
        141
        * np.minimum(crea / 0.7, 1) ** (-0.329)
        * np.maximum(crea / 0.7, 1) ** (-1.209)
        * 0.993**age
        * 1.018
    )
    male_egfr = (
        141
        * np.minimum(crea / 0.9, 1) ** (-0.411)
        * np.maximum(crea / 0.9, 1) ** (-1.209)
        * 0.993**age
    )
    egfr = np.full_like(crea, np.nan, dtype=np.float64)
    egfr[sex == 2] = female_egfr[sex == 2]
    egfr[sex == 1] = male_egfr[sex == 1]
    return egfr


def kidney_creatinine_thresholds(source: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    age = source["age"][indices]
    sex = source["sex"][indices]
    thresholds = np.full(len(indices), np.nan, dtype=np.float64)
    valid = (age > 18) & np.isfinite(age) & np.isfinite(sex) & ((sex == 1) | (sex == 2))
    for pos in np.flatnonzero(valid):
        lo, hi = 0.1, 20.0
        for _ in range(48):
            mid = (lo + hi) / 2.0
            egfr = egfr_from_creatinine(np.asarray([mid]), np.asarray([age[pos]]), np.asarray([sex[pos]]))[0]
            if egfr < 60:
                hi = mid
            else:
                lo = mid
        thresholds[pos] = hi
    return thresholds


def write_label_distribution_values(
    path: Path,
    *,
    disease: str,
    feature: str,
    reference: np.ndarray,
    model_all: np.ndarray,
    model_false: np.ndarray,
    model_true: np.ndarray,
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["disease", "feature", "distribution", "value"])
        writer.writeheader()
        for distribution, values in (
            ("data", reference),
            ("model_all", model_all),
            ("model_label_false", model_false),
            ("model_label_true", model_true),
        ):
            for value in values:
                writer.writerow(
                    {
                        "disease": disease,
                        "feature": feature,
                        "distribution": distribution,
                        "value": value,
                    }
                )


def shared_bins(values: list[np.ndarray]) -> np.ndarray:
    finite = np.concatenate([value[np.isfinite(value)] for value in values if len(value)])
    if finite.size == 0:
        return np.linspace(0.0, 1.0, 20)
    lo, hi = np.percentile(finite, [0.1, 99.9])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    return np.linspace(float(lo), float(hi), 36)


def value_axis_label(base: str, value_scale: str) -> str:
    if value_scale == "transformed":
        return f"{base} (Gaussian quantile z)"
    return base


def write_scatter_data(
    path: Path,
    disease: str,
    features: list[str],
    arrays: dict[str, Any],
    *,
    generated_key: str,
    generated_fieldname: str,
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["disease", "feature", "test_position", "true_value", generated_fieldname, "observed"],
        )
        writer.writeheader()
        for feature_idx, feature in enumerate(features):
            for idx, (true_value, generated_value, observed) in enumerate(
                zip(
                    arrays["true_values"][:, feature_idx],
                    arrays[generated_key][:, feature_idx],
                    arrays["target_num_mask"][:, feature_idx] > 0,
                )
            ):
                writer.writerow(
                    {
                        "disease": disease,
                        "feature": feature,
                        "test_position": idx,
                        "true_value": true_value,
                        generated_fieldname: generated_value,
                        "observed": bool(observed),
                    }
                )


def write_distribution_values(
    path: Path,
    *,
    disease: str,
    feature: str,
    generated_bins: list[tuple[str, str, np.ndarray]],
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["disease", "feature", "bin", "value"])
        writer.writeheader()
        for name, _, values in generated_bins:
            for value in values:
                writer.writerow({"disease": disease, "feature": feature, "bin": name, "value": value})


def write_single_sample_values(
    path: Path,
    *,
    disease: str,
    feature: str,
    selected_indices: np.ndarray,
    true_values: np.ndarray,
    generated_values: list[np.ndarray],
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["disease", "feature", "test_position", "true_value", "sample_index", "generated_value"],
        )
        writer.writeheader()
        for idx, values in zip(selected_indices, generated_values):
            for sample_idx, value in enumerate(values):
                writer.writerow(
                    {
                        "disease": disease,
                        "feature": feature,
                        "test_position": int(idx),
                        "true_value": float(true_values[idx]),
                        "sample_index": sample_idx,
                        "generated_value": value,
                    }
                )


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def stable_int(value: str) -> int:
    result = 0
    for char in value:
        result = (result * 33 + ord(char)) % 100000
    return result


if __name__ == "__main__":
    main()
