#!/usr/bin/env python
"""Compare hepatitis B/C masked-AE before runs with Optuna trial AUROCs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DISEASES = ["hepatitis_b", "hepatitis_c"]
DISPLAY = {"hepatitis_b": "Hepatitis B", "hepatitis_c": "Hepatitis C"}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trial_rows = []
    before_rows = []
    for disease in DISEASES:
        trials = pd.read_csv(args.optuna_root / disease / "trials.csv")
        for row in trials.itertuples(index=False):
            trial_rows.append(
                {
                    "disease": disease,
                    "disease_label": DISPLAY[disease],
                    "kind": "optuna_trial",
                    "label": f"trial_{int(row.number):04d}",
                    "trial": int(row.number),
                    "valid_auroc": float(row.value),
                    "test_auroc": float(row.user_attrs_metric_test_auroc),
                    "best_step": int(row.user_attrs_metric_best_step),
                    "head_architecture": row.params_head_architecture,
                    "freeze_backbone": bool(row.params_freeze_backbone),
                    "class_weight": row.params_class_weight,
                    "learning_rate": float(row.params_learning_rate),
                    "batch_size": int(row.params_batch_size),
                }
            )
        for label, root in before_roots(args, disease):
            summary = read_json(root / disease / "summary.json")
            before_rows.append(
                {
                    "disease": disease,
                    "disease_label": DISPLAY[disease],
                    "kind": "before_optuna",
                    "label": label,
                    "trial": "",
                    "valid_auroc": float(summary["best_valid_auroc"]),
                    "test_auroc": float(summary["test_auroc"]),
                    "best_step": int(summary["best_step"]),
                    "head_architecture": "h256_256_fixed",
                    "freeze_backbone": False,
                    "class_weight": "balanced",
                    "learning_rate": 5e-4,
                    "batch_size": 4096,
                }
            )

    trial_table = pd.DataFrame(trial_rows)
    before_table = pd.DataFrame(before_rows)
    combined = pd.concat([trial_table, before_table], ignore_index=True)
    combined.to_csv(args.output_dir / "masked_ae_hepatitis_bc_trial_auroc.csv", index=False)

    summary = summarize(trial_table, before_table)
    summary.to_csv(args.output_dir / "masked_ae_hepatitis_bc_trial_auroc_summary.csv", index=False)

    plot_trial_comparison(trial_table, before_table, args.output_dir)


def before_roots(args: argparse.Namespace, disease: str) -> list[tuple[str, Path]]:
    roots = [("before_seed42", args.before_seed42_root)]
    if disease == "hepatitis_b":
        roots.append(("before_seed42_repeat", args.before_b_repeat_root))
    roots.append(("before_seed43", args.before_seed43_root))
    return roots


def summarize(trials: pd.DataFrame, before: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for disease in DISEASES:
        t = trials[trials["disease"].eq(disease)].copy()
        b = before[before["disease"].eq(disease)].copy()
        valid_best = t.sort_values("valid_auroc", ascending=False).iloc[0]
        test_best = t.sort_values("test_auroc", ascending=False).iloc[0]
        rows.append(
            {
                "disease": disease,
                "optuna_valid_best_trial": int(valid_best["trial"]),
                "optuna_valid_best_valid_auroc": float(valid_best["valid_auroc"]),
                "optuna_valid_best_test_auroc": float(valid_best["test_auroc"]),
                "optuna_test_best_trial": int(test_best["trial"]),
                "optuna_test_best_valid_auroc": float(test_best["valid_auroc"]),
                "optuna_test_best_test_auroc": float(test_best["test_auroc"]),
                "optuna_test_auroc_mean": float(t["test_auroc"].mean()),
                "optuna_test_auroc_min": float(t["test_auroc"].min()),
                "optuna_test_auroc_max": float(t["test_auroc"].max()),
                "before_test_auroc_mean": float(b["test_auroc"].mean()),
                "before_test_auroc_min": float(b["test_auroc"].min()),
                "before_test_auroc_max": float(b["test_auroc"].max()),
            }
        )
    return pd.DataFrame(rows)


def plot_trial_comparison(trials: pd.DataFrame, before: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), sharey=True)
    before_colors = {
        "before_seed42": "#D97706",
        "before_seed42_repeat": "#92400E",
        "before_seed43": "#B45309",
    }
    for ax, disease in zip(axes, DISEASES, strict=True):
        t = trials[trials["disease"].eq(disease)].sort_values("trial")
        b = before[before["disease"].eq(disease)]
        ax.plot(t["trial"], t["test_auroc"], marker="o", color="#2563EB", label="Optuna trial test AUROC")
        ax.plot(
            t["trial"],
            t["valid_auroc"],
            marker=".",
            linestyle="--",
            color="#93C5FD",
            label="Optuna trial valid AUROC",
        )
        valid_best = t.loc[t["valid_auroc"].idxmax()]
        ax.scatter(
            [valid_best["trial"]],
            [valid_best["test_auroc"]],
            s=95,
            facecolors="none",
            edgecolors="#111827",
            linewidths=1.5,
            label="valid-best selected trial",
            zorder=5,
        )
        for row in b.itertuples(index=False):
            ax.axhline(
                row.test_auroc,
                color=before_colors[row.label],
                linewidth=1.6,
                alpha=0.9,
                label=f"{row.label} test AUROC",
            )
            ax.text(
                19.4,
                row.test_auroc,
                f"{row.test_auroc:.3f}",
                va="center",
                ha="left",
                color=before_colors[row.label],
                fontsize=9,
            )
        ax.set_title(DISPLAY[disease])
        ax.set_xlabel("Optuna trial number")
        ax.set_xlim(-0.7, 21.0)
        ax.set_ylim(0.45, 0.80)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("AUROC")
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        ncols=3,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("Masked-AE Hepatitis B/C: Before Runs vs Optuna Trials", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.25, wspace=0.12)
    fig.savefig(output_dir / "masked_ae_hepatitis_bc_trial_auroc.png", dpi=220)
    fig.savefig(output_dir / "masked_ae_hepatitis_bc_trial_auroc.pdf")
    plt.close(fig)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optuna-root",
        type=Path,
        default=Path("outputs/optuna/optuna_masked_ae_downstream_mask050_20t_8d_8gpu_20260528_142945"),
    )
    parser.add_argument(
        "--before-seed42-root",
        type=Path,
        default=Path("outputs/masked_ae_downstream/masked_ae_downstream_mask050_8diseases_before_optuna_20260528_180801"),
    )
    parser.add_argument(
        "--before-b-repeat-root",
        type=Path,
        default=Path("outputs/masked_ae_downstream/masked_ae_downstream_mask050_hepatitis_b_before_optuna_repeat_20260528_182431"),
    )
    parser.add_argument(
        "--before-seed43-root",
        type=Path,
        default=Path("outputs/masked_ae_downstream/masked_ae_downstream_mask050_hepatitis_bc_before_optuna_seed43_20260529_003209"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/masked_ae_hepatitis_bc_trial_auroc"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
