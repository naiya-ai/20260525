"""Plot disease-wise best CVAE AUROC against weighted CatBoost AUROC."""

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
    parser.add_argument("--cvae-metrics", type=Path, required=True)
    parser.add_argument("--catboost-summary", type=Path, required=True)
    parser.add_argument("--dataset-name", default="harmonized_knhanes_1998_2024")
    parser.add_argument("--output-dir", type=Path, required=True)
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

    cvae = pd.read_csv(args.cvae_metrics)
    catboost = pd.read_csv(args.catboost_summary)
    catboost = catboost[catboost["dataset_name"].eq(args.dataset_name)].copy()

    cvae_best = (
        cvae.sort_values(["target_group", "auroc_max", "usable_rate"], ascending=[True, False, False])
        .groupby("target_group", as_index=False)
        .head(1)
        .rename(
            columns={
                "target_group": "disease",
                "beta": "cvae_best_beta",
                "auroc_max": "cvae_best_auroc",
                "auroc_mean": "cvae_mean_auroc_at_best_beta",
                "n_usable": "cvae_n_usable",
                "n_total": "cvae_n_total",
            }
        )
    )
    boost = catboost.rename(columns={"target_group": "disease", "auroc": "catboost_auroc"})[
        ["disease", "catboost_auroc"]
    ]
    combined = cvae_best.merge(boost, on="disease", how="outer")
    combined["delta_cvae_minus_catboost"] = combined["cvae_best_auroc"] - combined["catboost_auroc"]
    combined["disease"] = pd.Categorical(combined["disease"], categories=DISEASE_ORDER, ordered=True)
    combined = combined.sort_values("disease")
    combined["cvae_best_beta_label"] = combined["cvae_best_beta"].map(beta_label)
    combined_path = args.output_dir / "cvae_best_vs_catboost_auroc.csv"
    combined.to_csv(combined_path, index=False)

    plot = combined.dropna(subset=["disease"]).copy()
    x = range(len(plot))
    width = 0.36

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
        plot["cvae_best_auroc"],
        width=width,
        label="CVAE best run",
        color="#F58518",
    )

    for i, row in enumerate(plot.itertuples(index=False)):
        if pd.notna(row.cvae_best_auroc):
            ax.text(
                i + width / 2,
                row.cvae_best_auroc + 0.008,
                f"b={row.cvae_best_beta_label}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#6B3A00",
            )
        if pd.notna(row.delta_cvae_minus_catboost):
            delta = row.delta_cvae_minus_catboost
            y = max(row.catboost_auroc, row.cvae_best_auroc) + 0.045
            color = "#166534" if delta >= 0 else "#991B1B"
            ax.text(i, y, f"{delta:+.3f}", ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")

    ax.set_title(
        f"AUROC comparison by disease: CVAE best run vs weighted CatBoost ({args.dataset_name})",
        fontsize=17,
        fontweight="bold",
        pad=18,
    )
    ax.set_ylabel("Test AUROC", fontsize=12)
    ax.set_ylim(0.45, 0.96)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(value).replace("_", "\n") for value in plot["disease"]], fontsize=10)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    ax.text(
        0.01,
        -0.16,
        "CVAE best uses max AUROC across beta candidates and 5 repeats; label above CVAE bar is the beta of that best run. Delta is CVAE - CatBoost.",
        transform=ax.transAxes,
        fontsize=9,
        color="#6B7280",
    )

    fig.tight_layout(rect=[0.02, 0.07, 0.99, 0.96])
    png_path = args.output_dir / "cvae_best_vs_catboost_auroc.png"
    svg_path = args.output_dir / "cvae_best_vs_catboost_auroc.svg"
    fig.savefig(png_path, facecolor="white")
    fig.savefig(svg_path, facecolor="white")
    plt.close(fig)

    print(f"wrote {combined_path}")
    print(f"wrote {png_path}")
    print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()
