#!/usr/bin/env python
"""Plot CatBoost vs masked-AE downstream before/after Optuna."""

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
    "hepatitis_b",
    "hepatitis_c",
    "kidney_disease",
    "anemia",
]

DISPLAY_NAMES = {
    "diabetes": "Diabetes",
    "hypertension": "Hypertension",
    "dyslipidemia": "Dyslipidemia",
    "liver_disease": "Liver disease",
    "hepatitis_b": "Hepatitis B",
    "hepatitis_c": "Hepatitis C",
    "kidney_disease": "Kidney disease",
    "anemia": "Anemia",
}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for disease in DISEASES:
        catboost = read_json(
            args.catboost_root / f"weighted_catboost_{disease}_metrics.json"
        )
        before = read_json(args.before_root / disease / "summary.json")
        after = pd.read_csv(args.optuna_root / disease / "top_trials.csv").iloc[0]

        rows.extend(
            [
                {
                    "disease": disease,
                    "disease_label": DISPLAY_NAMES[disease],
                    "model": "CatBoost",
                    "test_auroc": float(catboost["auroc"]),
                    "selection_metric": "",
                    "best_step": "",
                    "trial": "",
                },
                {
                    "disease": disease,
                    "disease_label": DISPLAY_NAMES[disease],
                    "model": "MAE before Optuna",
                    "test_auroc": float(before["test_auroc"]),
                    "selection_metric": float(before["best_valid_auroc"]),
                    "best_step": int(before["best_step"]),
                    "trial": "",
                },
                {
                    "disease": disease,
                    "disease_label": DISPLAY_NAMES[disease],
                    "model": "MAE after Optuna",
                    "test_auroc": float(after["user_attrs_metric_test_auroc"]),
                    "selection_metric": float(after["value"]),
                    "best_step": int(after["user_attrs_metric_best_step"]),
                    "trial": int(after["number"]),
                },
            ]
        )

    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "catboost_mae_optuna_test_auroc.csv", index=False)
    wide = table.pivot(index="disease_label", columns="model", values="test_auroc").loc[
        [DISPLAY_NAMES[disease] for disease in DISEASES]
    ]
    wide.to_csv(args.output_dir / "catboost_mae_optuna_test_auroc_wide.csv")

    plot_grouped_bars(table, args.output_dir)
    plot_delta(wide, args.output_dir)


def plot_grouped_bars(table: pd.DataFrame, output_dir: Path) -> None:
    model_order = ["CatBoost", "MAE before Optuna", "MAE after Optuna"]
    colors = {
        "CatBoost": "#4B5563",
        "MAE before Optuna": "#D97706",
        "MAE after Optuna": "#2563EB",
    }
    labels = [DISPLAY_NAMES[disease] for disease in DISEASES]
    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    for offset, model in zip([-width, 0.0, width], model_order, strict=True):
        values = [
            float(
                table.loc[
                    (table["disease_label"] == label) & (table["model"] == model),
                    "test_auroc",
                ].iloc[0]
            )
            for label in labels
        ]
        bars = ax.bar(x + offset, values, width=width, label=model, color=colors[model])
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], fontsize=8, padding=2)

    ax.set_ylabel("Test AUROC")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncols=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.set_title("CatBoost vs Masked-AE Downstream Before and After Optuna", pad=16)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.28)

    fig.savefig(output_dir / "catboost_mae_optuna_test_auroc.png", dpi=220)
    fig.savefig(output_dir / "catboost_mae_optuna_test_auroc.pdf")
    plt.close(fig)


def plot_delta(wide: pd.DataFrame, output_dir: Path) -> None:
    delta = wide["MAE after Optuna"] - wide["MAE before Optuna"]
    colors = np.where(delta >= 0, "#2563EB", "#B91C1C")
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    bars = ax.bar(delta.index, delta.values, color=colors)
    ax.axhline(0.0, color="#111827", linewidth=1)
    ax.bar_label(bars, labels=[f"{value:+.3f}" for value in delta.values], fontsize=9, padding=3)
    ax.set_ylabel("Test AUROC delta")
    ax.set_xticks(np.arange(len(delta.index)))
    ax.set_xticklabels(delta.index, rotation=25, ha="right")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Optuna Gain Over Fixed Masked-AE Downstream")
    fig.savefig(output_dir / "mae_optuna_delta_test_auroc.png", dpi=220)
    fig.savefig(output_dir / "mae_optuna_delta_test_auroc.pdf")
    plt.close(fig)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catboost-root",
        type=Path,
        default=Path(
            "outputs/catboost_weighted_range_compare/20260525_071919/"
            "harmonized_knhanes_1998_2024/weighted_catboost"
        ),
    )
    parser.add_argument(
        "--before-root",
        type=Path,
        default=Path(
            "outputs/masked_ae_downstream/"
            "masked_ae_downstream_mask050_8diseases_before_optuna_20260528_180801"
        ),
    )
    parser.add_argument(
        "--optuna-root",
        type=Path,
        default=Path(
            "outputs/optuna/"
            "optuna_masked_ae_downstream_mask050_20t_8d_8gpu_20260528_142945"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/catboost_mae_optuna_comparison_8diseases"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
