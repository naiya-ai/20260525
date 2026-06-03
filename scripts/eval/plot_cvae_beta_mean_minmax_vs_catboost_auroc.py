"""Plot CVAE beta AUROC mean/min-max against weighted CatBoost AUROC."""

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
    parser.add_argument("--catboost-summary", type=Path, required=True)
    parser.add_argument("--dataset-name", default="harmonized_knhanes_1998_2024")
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def beta_tag(value: float) -> str:
    if value < 0.01:
        return f"{value:.3f}"
    if value < 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.1f}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = pd.read_csv(args.diagnostics_summary)
    cvae = diagnostics[diagnostics["beta"].astype(float).round(6).eq(round(args.beta, 6))].copy()

    # For beta=0.001 all runs were usable in the current sweep. The KL/activity
    # filter keeps the definition aligned with the beta-selection report for numeric targets.
    cvae["usable"] = cvae["test_auroc"].notna()
    cvae_agg = (
        cvae[cvae["usable"]]
        .groupby("target_group", as_index=False)
        .agg(
            cvae_auroc_mean=("test_auroc", "mean"),
            cvae_auroc_min=("test_auroc", "min"),
            cvae_auroc_max=("test_auroc", "max"),
            cvae_n=("test_auroc", "size"),
        )
        .rename(columns={"target_group": "disease"})
    )

    catboost = pd.read_csv(args.catboost_summary)
    catboost = catboost[catboost["dataset_name"].eq(args.dataset_name)].copy()
    boost = catboost.rename(columns={"target_group": "disease", "auroc": "catboost_auroc"})[
        ["disease", "catboost_auroc"]
    ]

    combined = cvae_agg.merge(boost, on="disease", how="outer")
    combined["delta_mean_cvae_minus_catboost"] = combined["cvae_auroc_mean"] - combined["catboost_auroc"]
    combined["disease"] = pd.Categorical(combined["disease"], categories=DISEASE_ORDER, ordered=True)
    combined = combined.sort_values("disease")

    csv_path = args.output_dir / f"cvae_beta{int(round(args.beta * 1000)):03d}_mean_minmax_vs_catboost_auroc.csv"
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
        yerr=[yerr_lower, yerr_upper],
        capsize=4,
        label=f"CVAE beta={beta_tag(args.beta)} mean [min-max]",
        color="#F58518",
        ecolor="#7C2D12",
        linewidth=0.8,
    )

    for i, row in enumerate(plot.itertuples(index=False)):
        delta = row.delta_mean_cvae_minus_catboost
        if pd.notna(delta):
            y = max(row.catboost_auroc, row.cvae_auroc_max) + 0.035
            color = "#166534" if delta >= 0 else "#991B1B"
            ax.text(i, y, f"{delta:+.3f}", ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")
        if pd.notna(row.cvae_n):
            ax.text(
                i + width / 2,
                row.cvae_auroc_min - 0.025,
                f"n={int(row.cvae_n)}",
                ha="center",
                va="top",
                fontsize=8,
                color="#7C2D12",
            )

    ax.set_title(
        f"AUROC comparison by disease: CVAE beta={beta_tag(args.beta)} mean/min-max vs weighted CatBoost",
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
        "CVAE bar is mean AUROC across usable runs; error bar spans min to max. Delta is CVAE mean - CatBoost.",
        transform=ax.transAxes,
        fontsize=9,
        color="#6B7280",
    )

    fig.tight_layout(rect=[0.02, 0.07, 0.99, 0.96])
    png_path = args.output_dir / f"cvae_beta{int(round(args.beta * 1000)):03d}_mean_minmax_vs_catboost_auroc.png"
    svg_path = args.output_dir / f"cvae_beta{int(round(args.beta * 1000)):03d}_mean_minmax_vs_catboost_auroc.svg"
    fig.savefig(png_path, facecolor="white")
    fig.savefig(svg_path, facecolor="white")
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")
    print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()
