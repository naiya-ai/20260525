"""Plot selected-beta non-collapsed CVAE AUROC against weighted CatBoost AUROC."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DISEASE_ORDER = [
    "diabetes",
    "hypertension",
    "dyslipidemia",
    "liver_disease",
    "hepatitis_b",
    "hepatitis_c",
    "kidney_disease",
    "anemia",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-summary", type=Path, required=True)
    parser.add_argument("--selected-betas", type=Path)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--exclude-target-groups", nargs="*", default=[])
    parser.add_argument(
        "--protocol-auroc-for-categorical",
        action="store_true",
        help="Report categorical target AUROC by the evaluation protocol even if KL collapse diagnostics exclude all runs.",
    )
    parser.add_argument("--catboost-summary", type=Path, required=True)
    parser.add_argument("--dataset-name", default="harmonized_knhanes_1998_2024")
    parser.add_argument("--min-kl-mean", type=float, default=0.05)
    parser.add_argument("--min-active-units", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="selected_cvae_noncollapsed_vs_catboost_auroc")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def beta_label(value: float) -> str:
    if value < 0.01:
        return f"{value:.3f}"
    if value < 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.1f}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = pd.read_csv(args.diagnostics_summary)
    diagnostics["beta"] = diagnostics["beta"].astype(float)
    diagnostics = diagnostics[~diagnostics["target_group"].isin(args.exclude_target_groups)].copy()
    if args.beta is not None:
        selected = pd.DataFrame(
            {
                "target_group": sorted(diagnostics["target_group"].unique()),
                "beta": float(args.beta),
            }
        )
    elif args.selected_betas is not None:
        selected = pd.read_csv(args.selected_betas)[["target_group", "beta"]].copy()
        selected = selected[~selected["target_group"].isin(args.exclude_target_groups)].copy()
    else:
        raise ValueError("Either --selected-betas or --beta must be provided.")
    selected["beta"] = selected["beta"].astype(float)

    active_col = "test_active_units_kl_gt_0_001"
    if active_col not in diagnostics.columns:
        active_col = "test_active_units_mu_var_gt_0_01"
    diagnostics["noncollapsed"] = (
        diagnostics["test_auroc"].notna()
        & diagnostics["test_kl_mean"].ge(args.min_kl_mean)
        & diagnostics[active_col].fillna(0).ge(args.min_active_units)
    )

    selected_runs = diagnostics.merge(selected, on=["target_group", "beta"], how="inner")
    selected_runs["categorical_target"] = False
    if "test_prior_transformed_mae" in selected_runs.columns and "test_prior_cat_bce" in selected_runs.columns:
        selected_runs["categorical_target"] = (
            selected_runs["test_prior_transformed_mae"].isna()
            & selected_runs["test_prior_cat_bce"].notna()
        )
    selected_runs["auroc_included"] = selected_runs["noncollapsed"]
    if args.protocol_auroc_for_categorical:
        selected_runs["auroc_included"] = (
            selected_runs["auroc_included"]
            | (selected_runs["categorical_target"] & selected_runs["test_auroc"].notna())
        )
    usable = selected_runs[selected_runs["auroc_included"]].copy()
    cvae_agg = (
        usable.groupby("target_group", as_index=False)
        .agg(
            cvae_auroc_mean=("test_auroc", "mean"),
            cvae_auroc_min=("test_auroc", "min"),
            cvae_auroc_max=("test_auroc", "max"),
            cvae_n_used=("test_auroc", "size"),
        )
        .rename(columns={"target_group": "disease"})
    )
    selected_counts = (
        selected_runs.groupby(["target_group", "beta"], as_index=False)
        .agg(
            cvae_n_total=("test_auroc", "size"),
            cvae_n_noncollapsed=("noncollapsed", "sum"),
            cvae_n_auroc=("auroc_included", "sum"),
            categorical_target=("categorical_target", "max"),
        )
        .rename(columns={"target_group": "disease", "beta": "selected_beta"})
    )

    catboost = pd.read_csv(args.catboost_summary)
    boost = catboost[catboost["dataset_name"].eq(args.dataset_name)].rename(
        columns={"target_group": "disease", "auroc": "catboost_auroc"}
    )[["disease", "catboost_auroc"]]
    boost = boost[~boost["disease"].isin(args.exclude_target_groups)].copy()

    combined = selected_counts.merge(cvae_agg, on="disease", how="left").merge(boost, on="disease", how="outer")
    combined["delta_mean_cvae_minus_catboost"] = combined["cvae_auroc_mean"] - combined["catboost_auroc"]
    combined["disease"] = pd.Categorical(combined["disease"], categories=DISEASE_ORDER, ordered=True)
    combined = combined.sort_values("disease")

    csv_path = args.output_dir / f"{args.output_prefix}.csv"
    combined.to_csv(csv_path, index=False)

    plot = combined.dropna(subset=["disease"]).copy()
    x = list(range(len(plot)))
    width = 0.36
    yerr_lower = plot["cvae_auroc_mean"] - plot["cvae_auroc_min"]
    yerr_upper = plot["cvae_auroc_max"] - plot["cvae_auroc_mean"]

    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=args.dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.bar(
        [i - width / 2 for i in x],
        plot["catboost_auroc"],
        width=width,
        label="Weighted CatBoost",
        color="#4C78A8",
    )
    ax.bar(
        [i + width / 2 for i in x],
        plot["cvae_auroc_mean"],
        width=width,
        yerr=[yerr_lower.fillna(0), yerr_upper.fillna(0)],
        capsize=4,
        label="CVAE mean [min-max]",
        color="#F58518",
        ecolor="#7C2D12",
        linewidth=0.8,
    )

    baseline_y = 0.462
    for i, row in enumerate(plot.itertuples(index=False)):
        beta = beta_label(float(row.selected_beta)) if pd.notna(row.selected_beta) else "-"
        if pd.notna(row.cvae_auroc_mean):
            y = row.cvae_auroc_min - 0.025
            if bool(row.categorical_target) and int(row.cvae_n_noncollapsed) != int(row.cvae_n_used):
                n_label = f"AUROC n={int(row.cvae_n_used)}\nKL {int(row.cvae_n_noncollapsed)}/{int(row.cvae_n_total)}"
            else:
                n_label = f"n={int(row.cvae_n_used)}"
            ax.text(i + width / 2, y, f"b={beta}\n{n_label}", ha="center", va="top", fontsize=8, color="#7C2D12")
            delta = row.delta_mean_cvae_minus_catboost
            y_delta = max(row.catboost_auroc, row.cvae_auroc_max) + 0.035
            color = "#166534" if delta >= 0 else "#991B1B"
            ax.text(i, y_delta, f"{delta:+.3f}", ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")
        else:
            ax.text(
                i + width / 2,
                baseline_y,
                f"b={beta}\n0/{int(row.cvae_n_total)}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#991B1B",
                fontweight="bold",
            )

    ax.set_title(
        "AUROC comparison by disease: CVAE vs weighted CatBoost",
        fontsize=17,
        fontweight="bold",
        pad=18,
    )
    ax.set_ylabel("Test AUROC", fontsize=12)
    ax.set_ylim(0.45, 0.96)
    ax.set_xticks(x)
    ax.set_xticklabels([str(value).replace("_", "\n") for value in plot["disease"]], fontsize=10)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    ax.text(
        0.01,
        -0.16,
        f"Numeric CVAE excludes runs with KL < {args.min_kl_mean:g} or active units < {args.min_active_units}. "
        "With categorical protocol enabled, hepatitis AUROC uses softmax-positive probabilities even if KL diagnostics collapse. "
        "Delta is CVAE mean - CatBoost.",
        transform=ax.transAxes,
        fontsize=9,
        color="#6B7280",
    )

    fig.tight_layout(rect=[0.02, 0.07, 0.99, 0.96])
    png_path = args.output_dir / f"{args.output_prefix}.png"
    svg_path = args.output_dir / f"{args.output_prefix}.svg"
    fig.savefig(png_path, facecolor="white")
    fig.savefig(svg_path, facecolor="white")
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")
    print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()
