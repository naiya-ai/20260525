"""Plot best CVAE real-value indicator distributions from cached prior samples."""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from eval.cvae_common import (
    decode_source_context,
    disease_predictions,
    inverse_gaussian_quantile,
    load_checkpoint_config_and_schema,
    read_target_metadata,
)
from train.dataset import create_grouped_cvae_dataloader, load_grouped_cvae_schema


DISEASES = ("diabetes", "hypertension", "dyslipidemia", "liver_disease", "kidney_disease", "anemia")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-summary", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--selected-runs", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", default="datasets/preprocessed/gaussian_quantile")
    parser.add_argument("--dataset-name", default="harmonized_knhanes_1998_2024")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--diseases", nargs="+", default=list(DISEASES))
    parser.add_argument("--min-kl-mean", type=float, default=0.05)
    parser.add_argument("--min-active-units", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--save-raw-values",
        action="store_true",
        help="Also write per-sample distribution values. This can create very large CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    helper = load_distribution_helpers()
    best = select_best_runs(args)
    best.to_csv(args.output_dir / "best_beta010_noncollapsed_runs.csv", index=False)

    rows: list[dict[str, Any]] = []
    for row in best.itertuples(index=False):
        disease = str(row.target_group)
        beta_tag = str(row.beta_tag).zfill(3) if str(row.beta_tag) != "1000" else "1000"
        rep = int(row.rep)
        checkpoint = Path(row.checkpoint)
        cache_path = args.cache_dir / f"beta{beta_tag}_rep{rep}_{disease}_test_prior_samples.npz"
        if not cache_path.exists():
            raise FileNotFoundError(cache_path)
        disease_rows = plot_disease_from_cache(
            args=args,
            helper=helper,
            disease=disease,
            checkpoint=checkpoint,
            cache_path=cache_path,
            beta=float(row.beta),
            rep=rep,
            auroc=float(row.test_auroc),
        )
        rows.extend(disease_rows)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "cached_distribution_summary.csv", index=False)
    summary.to_csv(args.figures_dir / "cached_distribution_summary.csv", index=False)
    best.to_csv(args.figures_dir / "best_beta010_noncollapsed_runs.csv", index=False)
    print(f"wrote {args.output_dir}")
    print(f"wrote {args.figures_dir}")


def load_distribution_helpers():
    path = Path("scripts/eval/analyze_cvae_indicator_distributions.py")
    spec = importlib.util.spec_from_file_location("indicator_distribution_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_best_runs(args: argparse.Namespace) -> pd.DataFrame:
    if args.selected_runs is not None:
        selected = pd.read_csv(args.selected_runs)
        required = {"target_group", "beta", "beta_tag", "rep", "checkpoint", "test_auroc"}
        missing = required - set(selected.columns)
        if missing:
            raise ValueError(f"--selected-runs is missing columns: {sorted(missing)}")
        selected = selected[selected["target_group"].isin(args.diseases)].copy()
        selected["beta"] = selected["beta"].astype(float)
        return selected.sort_values("target_group")

    diagnostics = pd.read_csv(args.diagnostics_summary)
    diagnostics["beta"] = diagnostics["beta"].astype(float)
    active_col = "test_active_units_kl_gt_0_001"
    if active_col not in diagnostics.columns:
        active_col = "test_active_units_mu_var_gt_0_01"
    diagnostics["noncollapsed"] = (
        diagnostics["test_auroc"].notna()
        & diagnostics["test_kl_mean"].ge(args.min_kl_mean)
        & diagnostics[active_col].fillna(0).ge(args.min_active_units)
    )
    subset = diagnostics[
        diagnostics["target_group"].isin(args.diseases)
        & diagnostics["beta"].round(6).eq(round(float(args.beta), 6))
        & diagnostics["noncollapsed"]
    ].copy()
    if subset.empty:
        raise ValueError(f"No non-collapsed runs found for beta={args.beta}.")
    best = (
        subset.sort_values(["target_group", "test_auroc"], ascending=[True, False])
        .groupby("target_group", as_index=False)
        .head(1)
        .sort_values("target_group")
    )
    missing = sorted(set(args.diseases) - set(best["target_group"]))
    if missing:
        raise ValueError(f"Missing non-collapsed beta={args.beta} best run for: {missing}")
    return best


def plot_disease_from_cache(
    *,
    args: argparse.Namespace,
    helper,
    disease: str,
    checkpoint: Path,
    cache_path: Path,
    beta: float,
    rep: int,
    auroc: float,
) -> list[dict[str, Any]]:
    config_args = argparse.Namespace(checkpoint=checkpoint)
    config, _ = load_checkpoint_config_and_schema(config_args)
    config["data"]["target_group"] = disease
    config["data"]["dataset_name"] = args.dataset_name
    config["data"]["dataset_root"] = args.dataset_root
    schema = load_grouped_cvae_schema(config)
    target_metadata = read_target_metadata(config)
    features = list(target_metadata["features"]["num"])
    if not features:
        return []

    source = load_source_context(config, schema, args)
    cache = np.load(cache_path, allow_pickle=True)
    generated = cache["generated_num"].astype(np.float32, copy=False)
    target_num = cache["target_num"].astype(np.float32, copy=False)
    target_mask = cache["target_num_mask"].astype(np.float32, copy=False) > 0
    num_samples = int(np.asarray(cache["num_samples"]).item())
    true_values = inverse_num_matrix(target_num, target_metadata)
    true_indicators = {feature: true_values[:, idx] for idx, feature in enumerate(features)}
    labels = np.asarray(disease_predictions(disease, true_indicators, source), dtype=object)

    rows: list[dict[str, Any]] = []
    disease_dir = args.output_dir / disease
    figure_disease_dir = args.figures_dir / "distribution_analysis" / disease
    disease_dir.mkdir(parents=True, exist_ok=True)
    figure_disease_dir.mkdir(parents=True, exist_ok=True)

    for feature_idx, feature in enumerate(features):
        observed = (target_mask[:, feature_idx] > 0) & np.isfinite(true_values[:, feature_idx])
        observed_indices = np.flatnonzero(observed)
        false_indices = np.asarray([idx for idx in observed_indices if labels[idx] is False], dtype=np.int64)
        true_indices = np.asarray([idx for idx in observed_indices if labels[idx] is True], dtype=np.int64)

        model_all = inverse_feature_samples(generated, target_metadata, feature_idx, observed_indices)
        model_false = inverse_feature_samples(generated, target_metadata, feature_idx, false_indices)
        model_true = inverse_feature_samples(generated, target_metadata, feature_idx, true_indices)
        reference = true_values[observed_indices, feature_idx]
        indicator_positive_probs = {
            "data": helper.indicator_positive_probability(
                disease=disease,
                feature=feature,
                values=reference,
                source=source,
                indices=observed_indices,
                num_samples=1,
            ),
            "model_all": helper.indicator_positive_probability(
                disease=disease,
                feature=feature,
                values=model_all,
                source=source,
                indices=observed_indices,
                num_samples=num_samples,
            ),
            "model_false": helper.indicator_positive_probability(
                disease=disease,
                feature=feature,
                values=model_false,
                source=source,
                indices=false_indices,
                num_samples=num_samples,
            ),
            "model_true": helper.indicator_positive_probability(
                disease=disease,
                feature=feature,
                values=model_true,
                source=source,
                indices=true_indices,
                num_samples=num_samples,
            ),
        }
        png_path = disease_dir / f"{disease}_{feature}_beta{beta:g}_rep{rep}_cached_distribution.png"
        plot_four_panel(
            path=png_path,
            helper=helper,
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
            beta=beta,
            rep=rep,
            auroc=auroc,
            cohort_label="all",
            dpi=args.dpi,
        )
        figure_path = figure_disease_dir / png_path.name
        figure_path.write_bytes(png_path.read_bytes())
        if args.save_raw_values:
            write_values(
                disease_dir / f"{disease}_{feature}_beta{beta:g}_rep{rep}_cached_distribution.csv",
                disease=disease,
                feature=feature,
                reference=reference,
                model_all=model_all,
                model_false=model_false,
                model_true=model_true,
            )
        rows.append(
            {
                "disease": disease,
                "feature": feature,
                "beta": beta,
                "rep": rep,
                "auroc": auroc,
                "num_samples_per_row": num_samples,
                "n_observed": int(len(observed_indices)),
                "n_label_false": int(len(false_indices)),
                "n_label_true": int(len(true_indices)),
                "data_indicator_positive_probability": indicator_positive_probs["data"],
                "model_all_indicator_positive_probability": indicator_positive_probs["model_all"],
                "model_false_indicator_positive_probability": indicator_positive_probs["model_false"],
                "model_true_indicator_positive_probability": indicator_positive_probs["model_true"],
                "figure": str(figure_path),
                "cohort": "all",
            }
        )

        if "sex" not in source:
            continue
        for sex_label, sex_code in (("male", 1), ("female", 2)):
            sex_observed = observed & (source["sex"] == sex_code)
            sex_indices = np.flatnonzero(sex_observed)
            if sex_indices.size == 0:
                continue
            sex_false_indices = np.asarray([idx for idx in sex_indices if labels[idx] is False], dtype=np.int64)
            sex_true_indices = np.asarray([idx for idx in sex_indices if labels[idx] is True], dtype=np.int64)
            sex_model_all = inverse_feature_samples(generated, target_metadata, feature_idx, sex_indices)
            sex_model_false = inverse_feature_samples(generated, target_metadata, feature_idx, sex_false_indices)
            sex_model_true = inverse_feature_samples(generated, target_metadata, feature_idx, sex_true_indices)
            sex_reference = true_values[sex_indices, feature_idx]
            sex_indicator_positive_probs = {
                "data": helper.indicator_positive_probability(
                    disease=disease,
                    feature=feature,
                    values=sex_reference,
                    source=source,
                    indices=sex_indices,
                    num_samples=1,
                ),
                "model_all": helper.indicator_positive_probability(
                    disease=disease,
                    feature=feature,
                    values=sex_model_all,
                    source=source,
                    indices=sex_indices,
                    num_samples=num_samples,
                ),
                "model_false": helper.indicator_positive_probability(
                    disease=disease,
                    feature=feature,
                    values=sex_model_false,
                    source=source,
                    indices=sex_false_indices,
                    num_samples=num_samples,
                ),
                "model_true": helper.indicator_positive_probability(
                    disease=disease,
                    feature=feature,
                    values=sex_model_true,
                    source=source,
                    indices=sex_true_indices,
                    num_samples=num_samples,
                ),
            }
            sex_dir = disease_dir / "by_sex" / sex_label
            sex_figure_dir = args.figures_dir / "distribution_analysis_by_sex" / sex_label / disease
            sex_dir.mkdir(parents=True, exist_ok=True)
            sex_figure_dir.mkdir(parents=True, exist_ok=True)
            sex_png_path = sex_dir / f"{disease}_{feature}_{sex_label}_beta{beta:g}_rep{rep}_cached_distribution.png"
            plot_four_panel(
                path=sex_png_path,
                helper=helper,
                disease=disease,
                feature=feature,
                reference=sex_reference,
                model_all=sex_model_all,
                model_false=sex_model_false,
                model_true=sex_model_true,
                source=source,
                observed_indices=sex_indices,
                n_observed=len(sex_indices),
                n_false=len(sex_false_indices),
                n_true=len(sex_true_indices),
                num_samples=num_samples,
                indicator_positive_probs=sex_indicator_positive_probs,
                beta=beta,
                rep=rep,
                auroc=auroc,
                cohort_label=sex_label,
                dpi=args.dpi,
            )
            sex_figure_path = sex_figure_dir / sex_png_path.name
            sex_figure_path.write_bytes(sex_png_path.read_bytes())
            rows.append(
                {
                    "disease": disease,
                    "feature": feature,
                    "beta": beta,
                    "rep": rep,
                    "auroc": auroc,
                    "num_samples_per_row": num_samples,
                    "n_observed": int(len(sex_indices)),
                    "n_label_false": int(len(sex_false_indices)),
                    "n_label_true": int(len(sex_true_indices)),
                    "data_indicator_positive_probability": sex_indicator_positive_probs["data"],
                    "model_all_indicator_positive_probability": sex_indicator_positive_probs["model_all"],
                    "model_false_indicator_positive_probability": sex_indicator_positive_probs["model_false"],
                    "model_true_indicator_positive_probability": sex_indicator_positive_probs["model_true"],
                    "figure": str(sex_figure_path),
                    "cohort": sex_label,
                }
            )
    return rows


def load_source_context(config: dict[str, Any], schema: dict[str, Any], args: argparse.Namespace) -> dict[str, np.ndarray]:
    loader = create_grouped_cvae_dataloader(
        config,
        split="test",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        seed=args.seed,
    )
    parts: dict[str, list[torch.Tensor]] = {
        "source_num": [],
        "source_cat": [],
        "source_num_mask": [],
        "source_cat_mask": [],
    }
    for batch in loader:
        for key in parts:
            parts[key].append(batch[key].detach().cpu())
    batch = {key: torch.cat(values, dim=0) for key, values in parts.items()}
    return decode_source_context(batch=batch, schema=schema, config=config)


def inverse_num_matrix(values: np.ndarray, target_metadata: dict[str, Any]) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    for idx, state in enumerate(target_metadata["gaussian_quantile"]["features"]):
        out[:, idx] = inverse_gaussian_quantile(values[:, idx], state)
    return out


def inverse_feature_samples(
    generated: np.ndarray,
    target_metadata: dict[str, Any],
    feature_idx: int,
    indices: np.ndarray,
) -> np.ndarray:
    if indices.size == 0:
        return np.asarray([], dtype=np.float64)
    state = target_metadata["gaussian_quantile"]["features"][feature_idx]
    values = generated[indices, :, feature_idx].reshape(-1)
    return inverse_gaussian_quantile(values, state)


def plot_four_panel(
    *,
    path: Path,
    helper,
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
    beta: float,
    rep: int,
    auroc: float,
    cohort_label: str,
    dpi: int,
) -> None:
    items = [
        ("data", f"Data distribution (test observed, n={n_observed})", reference),
        ("model_all", f"Model distribution (cached prior samples, N={num_samples}/row)", model_all),
        ("model_false", f"Model distribution | disease_label = false (n={n_false})", model_false),
        ("model_true", f"Model distribution | disease_label = true (n={n_true})", model_true),
    ]
    bins = helper.shared_bins([values for _, _, values in items])
    indication = helper.disease_indicator_indication(disease, feature, source, observed_indices)
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
        helper.draw_indication_region(ax, indication, float(bins[0]), float(bins[-1]))
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
    cohort_suffix = "" if cohort_label == "all" else f" / {cohort_label}"
    fig.suptitle(
        f"{disease} / {feature}{cohort_suffix}: beta={beta:g}, best rep={rep}, AUROC={auroc:.3f}",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def write_values(
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
                writer.writerow({"disease": disease, "feature": feature, "distribution": distribution, "value": value})


if __name__ == "__main__":
    main()
