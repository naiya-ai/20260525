"""Render train label counts for weighted CatBoost range comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


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


def main() -> None:
    args = parse_args()
    rows = read_summary(args.summary)
    plot_table(rows, args.output)
    print(f"wrote {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_summary(path: Path) -> dict[tuple[str, str], dict[str, int]]:
    rows: dict[tuple[str, str], dict[str, int]] = {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            dataset = row["dataset_name"]
            disease = row["target_group"]
            metrics = read_metrics(row)
            train_counts = metrics.get("train_label_counts", {})
            true_count = int(train_counts.get("true", 0))
            false_count = int(train_counts.get("false", 0))
            rows[(dataset, disease)] = {
                "true": true_count,
                "labeled": true_count + false_count,
            }
    return rows


def read_metrics(row: dict[str, str]) -> dict:
    metrics_path = row.get("metrics_path", "")
    if not metrics_path:
        return {}
    path = Path(metrics_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def plot_table(rows: dict[tuple[str, str], dict[str, int]], output: Path) -> None:
    diseases = [
        disease
        for disease in DISEASE_ORDER
        if any((dataset, disease) in rows for dataset in RANGE_ORDER)
    ]
    cell_text = []
    for dataset in RANGE_ORDER:
        row = []
        for disease in diseases:
            counts = rows.get((dataset, disease))
            if counts is None:
                row.append("NA")
            else:
                row.append(f"{counts['true']:,} / {counts['labeled']:,}")
        cell_text.append(row)

    fig_width = max(14.0, 1.45 * len(diseases) + 4.2)
    fig, ax = plt.subplots(figsize=(fig_width, 3.3))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=[RANGE_LABELS.get(dataset, dataset) for dataset in RANGE_ORDER],
        colLabels=[disease.replace("_", " ") for disease in diseases],
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.65)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        cell.set_linewidth(0.8)
        if row_idx == 0 or col_idx == -1:
            cell.set_facecolor("#f3f4f6")
            cell.set_text_props(weight="bold", color="#111827")
        else:
            cell.set_facecolor("white")
            cell.set_text_props(color="#111827")

    ax.set_title(
        "Train Label Counts by Dataset Range (true / labeled)",
        fontsize=13,
        fontweight="bold",
        pad=16,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
