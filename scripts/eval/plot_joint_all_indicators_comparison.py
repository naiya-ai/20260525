"""Plot all-indicator CVAE/DDPM AUROC against CatBoost per-disease baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DISEASES = ["diabetes", "hypertension", "dyslipidemia", "liver_disease", "hepatitis_b", "hepatitis_c", "kidney_disease", "anemia"]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.DataFrame({"disease": DISEASES})
    combined = combined.merge(load_catboost(args.catboost_summary, args.dataset_name), on="disease", how="left")
    if args.cvae_metrics is not None and args.cvae_metrics.exists():
        combined = combined.merge(load_model(args.cvae_metrics, "cvae"), on="disease", how="left")
    if args.ddpm_metrics is not None and args.ddpm_metrics.exists():
        combined = combined.merge(load_model(args.ddpm_metrics, "ddpm"), on="disease", how="left")
    csv_path = args.output_dir / f"{args.output_prefix}.csv"
    png_path = args.output_dir / f"{args.output_prefix}.png"
    combined.to_csv(csv_path, index=False)
    plot(combined, png_path, args)
    print(f"saved {csv_path}")
    print(f"saved {png_path}")


def load_catboost(path: Path, dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["dataset_name"].eq(dataset_name) & df["target_group"].isin(DISEASES)].copy()
    return df[["target_group", "auroc", "n_evaluable"]].rename(
        columns={"target_group": "disease", "auroc": "catboost_auroc", "n_evaluable": "catboost_n"}
    )


def load_model(path: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["disease", "auroc", "n_evaluable"]].rename(
        columns={"auroc": f"{prefix}_auroc", "n_evaluable": f"{prefix}_n"}
    )


def plot(df: pd.DataFrame, output_path: Path, args: argparse.Namespace) -> None:
    series = [("CatBoost", "catboost_auroc", "#4C78A8")]
    if "cvae_auroc" in df.columns:
        series.append(("CVAE all indicators", "cvae_auroc", "#F58518"))
    if "ddpm_auroc" in df.columns:
        series.append(("DDPM all indicators", "ddpm_auroc", "#54A24B"))
    x = np.arange(len(df))
    width = 0.8 / len(series)
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, len(series))
    fig, ax = plt.subplots(figsize=(14, 7.5), dpi=args.dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for offset, (label, column, color) in zip(offsets, series):
        ax.bar(x + offset, df[column], width=width, label=label, color=color)
        for xpos, value in zip(x + offset, df[column]):
            if pd.notna(value):
                ax.text(xpos, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("All-Indicator Target: Disease AUROC", fontsize=17, fontweight="bold")
    ax.set_ylabel("Test AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in df["disease"]], fontsize=9)
    ax.set_ylim(0.45, 1.0)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvae-metrics", type=Path)
    parser.add_argument("--ddpm-metrics", type=Path)
    parser.add_argument("--catboost-summary", type=Path, default=Path("outputs/catboost_weighted_range_compare/20260525_071919/weighted_catboost_range_summary.csv"))
    parser.add_argument("--dataset-name", default="harmonized_knhanes_1998_2024")
    parser.add_argument("--output-dir", type=Path, default=Path("figures/all_indicator_joint_comparison"))
    parser.add_argument("--output-prefix", default="all_indicator_joint_auroc")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


if __name__ == "__main__":
    main()
