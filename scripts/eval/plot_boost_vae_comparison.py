"""Plot CatBoost and original CVAE sensitivity/specificity."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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

VARIANT_ORDER = ["catboost", "cvae"]
VARIANT_LABELS = {
    "catboost": "CatBoost",
    "cvae": "Original CVAE",
}
VARIANT_COLORS = {
    "catboost": "#16a34a",
    "cvae": "#2563eb",
}


def main() -> None:
    args = parse_args()
    rows = read_summary(args.summary)
    plot(rows, args.output)
    print(f"wrote {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_summary(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    rows: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            sensitivity = parse_float(row["sensitivity"])
            specificity = parse_float(row["specificity"])
            rows[(row["variant"], row["disease"])] = {
                "sensitivity": sensitivity,
                "specificity": specificity,
                "balanced_accuracy": 0.5 * (sensitivity + specificity),
                "threshold": parse_float(row["threshold"]),
            }
    return rows


def parse_float(value: str) -> float:
    if value in {"", "None", "nan"}:
        return float("nan")
    return float(value)


def plot(rows: dict[tuple[str, str], dict[str, float]], output: Path) -> None:
    diseases = [
        disease
        for disease in DISEASE_ORDER
        if any((variant, disease) in rows for variant in VARIANT_ORDER)
    ]
    x = np.arange(len(diseases))
    width = 0.32
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        "CatBoost vs CVAE: Validation-Threshold Disease Evaluation",
        fontsize=16,
        fontweight="bold",
    )

    for ax, metric, ylabel in [
        (axes[0], "sensitivity", "Sensitivity"),
        (axes[1], "specificity", "Specificity"),
        (axes[2], "balanced_accuracy", "Balanced accuracy"),
    ]:
        for variant_idx, variant in enumerate(VARIANT_ORDER):
            values = [
                rows.get((variant, disease), {}).get(metric, np.nan)
                for disease in diseases
            ]
            bars = ax.bar(
                x + (variant_idx - 0.5) * width,
                values,
                width,
                label=VARIANT_LABELS[variant],
                color=VARIANT_COLORS[variant],
            )
            annotate_bars(ax, bars)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=False, ncols=2)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([disease.replace("_", " ") for disease in diseases], rotation=25, ha="right")
    axes[-1].set_xlabel("Disease")
    fig.text(
        0.01,
        0.01,
        "Thresholds are selected on validation balanced accuracy. "
        "CVAE probabilities are positive ratios over prior target samples.",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def annotate_bars(ax: plt.Axes, bars) -> None:
    for bar in bars:
        value = bar.get_height()
        if not np.isfinite(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.02, 1.03),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
            color="#111827",
        )


if __name__ == "__main__":
    main()
