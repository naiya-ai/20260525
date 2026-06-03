"""Plot DDPM AUROC against CatBoost and an optional CVAE comparison CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DISEASES = ["diabetes", "hypertension", "dyslipidemia", "liver_disease", "kidney_disease", "anemia"]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ddpm = load_ddpm(args.ddpm_eval_root)
    catboost = load_catboost(args.catboost_summary, args.dataset_name)
    combined = pd.DataFrame({"target_group": DISEASES}).merge(catboost, on="target_group", how="left")
    if args.cvae_comparison_csv is not None:
        combined = combined.merge(load_cvae(args.cvae_comparison_csv), on="target_group", how="left")
    combined = combined.merge(ddpm, on="target_group", how="left")
    csv_path = args.output_dir / f"{args.output_prefix}.csv"
    png_path = args.output_dir / f"{args.output_prefix}.png"
    combined.to_csv(csv_path, index=False)
    plot(combined, png_path, args)
    print(f"saved {csv_path}")
    print(f"saved {png_path}")


def load_ddpm(eval_root: Path) -> pd.DataFrame:
    rows = []
    for disease in DISEASES:
        path = eval_root / disease / f"{disease}_ddpm_metrics.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        rows.append(
            {
                "target_group": disease,
                "ddpm_auroc": payload.get("auroc"),
                "ddpm_n_evaluable": payload.get("n_evaluable"),
                "ddpm_best_valid_loss": payload.get("best_valid_loss"),
                "ddpm_checkpoint_step": payload.get("checkpoint_step"),
                "ddpm_num_samples": payload.get("num_samples"),
            }
        )
    return pd.DataFrame(rows)


def load_catboost(path: Path, dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df[df["dataset_name"].eq(dataset_name) & df["target_group"].isin(DISEASES)].copy()
    return out[["target_group", "auroc", "n_evaluable"]].rename(
        columns={"auroc": "catboost_auroc", "n_evaluable": "catboost_n_evaluable"}
    )


def load_cvae(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["target_group"].isin(DISEASES)].copy()
    if "eddi_auroc" in df.columns:
        return df[["target_group", "eddi_auroc"]].rename(columns={"eddi_auroc": "cvae_auroc"})
    if "cvae_auroc" in df.columns:
        return df[["target_group", "cvae_auroc"]]
    if "auroc" in df.columns:
        return df[["target_group", "auroc"]].rename(columns={"auroc": "cvae_auroc"})
    raise ValueError(f"Could not find a CVAE AUROC column in {path}")


def plot(df: pd.DataFrame, output_path: Path, args: argparse.Namespace) -> None:
    x = np.arange(len(df))
    has_cvae = "cvae_auroc" in df.columns
    series = [("Weighted CatBoost", "catboost_auroc", "#4C78A8")]
    if has_cvae:
        series.append(("CVAE", "cvae_auroc", "#F58518"))
    series.append(("DDPM", "ddpm_auroc", "#54A24B"))
    width = 0.8 / len(series)
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, len(series))
    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=args.dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for offset, (label, column, color) in zip(offsets, series):
        ax.bar(x + offset, df[column], width=width, label=label, color=color)
        for xpos, value in zip(x + offset, df[column]):
            if pd.notna(value):
                ax.text(xpos, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("AUROC Comparison: CatBoost vs CVAE vs DDPM", fontsize=17, fontweight="bold", pad=16)
    ax.set_ylabel("Test AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace("_", "\n") for name in df["target_group"]], fontsize=10)
    ax.set_ylim(0.45, 0.97)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddpm-eval-root", type=Path, required=True)
    parser.add_argument("--catboost-summary", type=Path, default=Path("outputs/catboost_weighted_range_compare/20260525_071919/weighted_catboost_range_summary.csv"))
    parser.add_argument("--dataset-name", default="harmonized_knhanes_1998_2024")
    parser.add_argument("--cvae-comparison-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/ddpm_auroc_comparison"))
    parser.add_argument("--output-prefix", default="ddpm_auroc_comparison")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


if __name__ == "__main__":
    main()
