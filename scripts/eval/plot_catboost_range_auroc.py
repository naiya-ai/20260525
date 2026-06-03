"""Plot weighted CatBoost AUROC across dataset ranges."""

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

RANGE_ORDER = [
    "harmonized_knhanes_2024",
    "harmonized_knhanes_2013_2024",
    "harmonized_knhanes_1998_2024",
    "harmonized_knhanes_1998_2024_plus_nhanes_1988_2023",
]

RANGE_LABELS = {
    "harmonized_knhanes_2024": "2024 KNHANES",
    "harmonized_knhanes_2013_2024": "2013-2024 KNHANES",
    "harmonized_knhanes_1998_2024": "1998-2024 KNHANES",
    "harmonized_knhanes_1998_2024_plus_nhanes_1988_2023": (
        "1998-2024 KNHANES\n+ 1988-2023 NHANES"
    ),
}

RANGE_COLORS = {
    "harmonized_knhanes_2024": "#2563eb",
    "harmonized_knhanes_2013_2024": "#16a34a",
    "harmonized_knhanes_1998_2024": "#f59e0b",
    "harmonized_knhanes_1998_2024_plus_nhanes_1988_2023": "#dc2626",
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
            dataset = row["dataset_name"]
            disease = row["target_group"]
            auroc = parse_float(row["auroc"])
            rows[(dataset, disease)] = {
                "auroc": auroc,
            }
    return rows


def parse_float(value) -> float:
    if value in {"", "None", "nan"}:
        return float("nan")
    return float(value)


def plot(rows: dict[tuple[str, str], dict[str, float]], output: Path) -> None:
    diseases = [
        disease
        for disease in DISEASE_ORDER
        if any((dataset, disease) in rows for dataset in RANGE_ORDER)
    ]
    x = np.arange(len(diseases))
    width = 0.18

    fig, ax = plt.subplots(figsize=(15, 7))
    for idx, dataset in enumerate(RANGE_ORDER):
        values = [
            rows.get((dataset, disease), {}).get("auroc", np.nan)
            for disease in diseases
        ]
        offset = (idx - (len(RANGE_ORDER) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=RANGE_LABELS.get(dataset, dataset),
            color=RANGE_COLORS.get(dataset),
        )
        annotate_bars(ax, bars)

    ax.set_title("Weighted CatBoost AUROC by Training Dataset Range", fontweight="bold")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.45, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([disease.replace("_", " ") for disease in diseases], rotation=25, ha="right")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=4, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def annotate_bars(ax: plt.Axes, bars) -> None:
    for bar in bars:
        value = bar.get_height()
        if not np.isfinite(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.008, 0.99),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
            color="#111827",
        )
if __name__ == "__main__":
    main()
