#!/usr/bin/env python
"""Plot 6-disease CVAE AUROC before/after Optuna, with CatBoost reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DISEASES = [
    "diabetes",
    "hypertension",
    "dyslipidemia",
    "liver_disease",
    "kidney_disease",
    "anemia",
]

DISPLAY_NAMES = {
    "diabetes": "Diabetes",
    "hypertension": "Hypertension",
    "dyslipidemia": "Dyslipidemia",
    "liver_disease": "Liver disease",
    "kidney_disease": "Kidney disease",
    "anemia": "Anemia",
}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    before = pd.read_csv(args.before_cvae_csv)
    for disease in DISEASES:
        best = read_json(args.optuna_root / disease / "best_trial.json")
        attrs = best["user_attrs"]
        catboost = read_json(args.catboost_root / f"weighted_catboost_{disease}_metrics.json")
        before_row = before.loc[before["disease"].eq(disease)].iloc[0]
        rows.append(
            {
                "disease": disease,
                "disease_label": DISPLAY_NAMES[disease],
                "catboost_auroc": float(catboost["auroc"]),
                "cvae_before_optuna_auroc": float(before_row["cvae_auroc_mean"]),
                "cvae_before_optuna_min": float(before_row["cvae_auroc_min"]),
                "cvae_before_optuna_max": float(before_row["cvae_auroc_max"]),
                "cvae_before_optuna_n_used": int(before_row["cvae_n_used"]),
                "cvae_optuna_auroc": float(attrs["metric_auroc"]),
                "cvae_objective": float(best["value"]),
                "cvae_best_trial": int(best["number"]),
                "cvae_cov90_abs_error": float(attrs["metric_cov90_abs_error"]),
                "cvae_coverage_ok": bool(attrs["metric_coverage_ok"]),
                "cvae_beta": float(best["params"]["beta"]),
                "cvae_latent_dim": int(best["params"]["latent_dim"]),
                "cvae_width_multiplier": float(best["params"]["width_multiplier"]),
            }
        )

    table = pd.DataFrame(rows)
    table["delta_cvae_minus_catboost"] = table["cvae_optuna_auroc"] - table["catboost_auroc"]
    table["delta_optuna_minus_before"] = (
        table["cvae_optuna_auroc"] - table["cvae_before_optuna_auroc"]
    )
    table.to_csv(args.output_dir / "cvae_optuna_6disease_auroc.csv", index=False)

    plot_auroc(table, args.output_dir, args.dpi)
    plot_delta(table, args.output_dir, args.dpi)


def plot_auroc(table: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    labels = table["disease_label"].tolist()
    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    cat = ax.bar(x - width, table["catboost_auroc"], width, label="Weighted CatBoost", color="#4B5563")
    before = ax.bar(
        x,
        table["cvae_before_optuna_auroc"],
        width,
        label="CVAE before Optuna",
        color="#D97706",
    )
    cvae = ax.bar(
        x + width,
        table["cvae_optuna_auroc"],
        width,
        label="CVAE after Optuna",
        color="#2563EB",
    )

    ax.bar_label(cat, labels=[f"{v:.3f}" for v in table["catboost_auroc"]], fontsize=8, padding=2)
    ax.bar_label(
        before,
        labels=[f"{v:.3f}" for v in table["cvae_before_optuna_auroc"]],
        fontsize=8,
        padding=2,
    )
    ax.bar_label(cvae, labels=[f"{v:.3f}" for v in table["cvae_optuna_auroc"]], fontsize=8, padding=2)

    ax.set_title("CVAE AUROC Before and After Optuna Across 6 Diseases", pad=16)
    ax.set_ylabel("Test AUROC")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncols=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.28)
    fig.savefig(output_dir / "cvae_optuna_6disease_auroc.png", dpi=dpi)
    fig.savefig(output_dir / "cvae_optuna_6disease_auroc.pdf")
    plt.close(fig)


def plot_delta(table: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    labels = table["disease_label"].tolist()
    delta = table["delta_optuna_minus_before"].to_numpy()
    x = np.arange(len(labels))
    colors = np.where(delta >= 0, "#2563EB", "#B91C1C")

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bars = ax.bar(x, delta, color=colors)
    ax.axhline(0, color="#111827", linewidth=1)
    ax.bar_label(bars, labels=[f"{v:+.3f}" for v in delta], fontsize=9, padding=3)
    ax.set_title("CVAE Optuna AUROC Minus CVAE Before Optuna", pad=16)
    ax.set_ylabel("Test AUROC delta")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.88, bottom=0.24)
    fig.savefig(output_dir / "cvae_optuna_6disease_delta_vs_before.png", dpi=dpi)
    fig.savefig(output_dir / "cvae_optuna_6disease_delta_vs_before.pdf")
    plt.close(fig)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optuna-root",
        type=Path,
        default=Path("outputs/optuna/cvae_without_disease_cov005_auroc_50t_4d_20260527_114059"),
    )
    parser.add_argument(
        "--catboost-root",
        type=Path,
        default=Path(
            "outputs/catboost_weighted_range_compare/20260525_071919/"
            "harmonized_knhanes_1998_2024/weighted_catboost"
        ),
    )
    parser.add_argument(
        "--before-cvae-csv",
        type=Path,
        default=Path(
            "outputs/repeats/cvae_z16_beta_selection_6diseases_noearly_20260525_234008/"
            "beta_selection_report_checkpoint_latest/"
            "beta010_cvae_noncollapsed_no_hepatitis_vs_catboost_auroc.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/cvae_optuna_6disease_auroc"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


if __name__ == "__main__":
    main()
