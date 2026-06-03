"""Summarize AUROC for repeated EDDI z2 beta experiments."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.root.glob("beta*/rep*/auroc.json")):
        data = json.loads(path.read_text())
        rows.append(
            {
                "beta": f"{float(data['beta']):.3f}",
                "rep": int(data["rep"]),
                "seed": int(data["seed"]),
                "auroc": f"{float(data['auroc']):.8f}",
                "n_evaluable": int(data["n_evaluable"]),
                "n_positive": int(data["n_positive"]),
                "run_dir": data["run_dir"],
                "probabilities": data["probabilities"],
            }
        )
    if not rows:
        raise ValueError(f"No AUROC JSON files found under {args.root}")

    detail_path = Path(args.detail_output)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    with detail_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for beta in sorted({row["beta"] for row in rows}, key=float):
        values = [float(row["auroc"]) for row in rows if row["beta"] == beta]
        summary_rows.append(
            {
                "beta": beta,
                "n_runs": len(values),
                "mean_auroc": f"{statistics.fmean(values):.8f}",
                "std_auroc": f"{statistics.stdev(values):.8f}" if len(values) > 1 else "0.00000000",
                "min_auroc": f"{min(values):.8f}",
                "max_auroc": f"{max(values):.8f}",
            }
        )

    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(summary_path)
    print(detail_path)
    for row in summary_rows:
        print(
            f"beta={row['beta']} n={row['n_runs']} "
            f"mean={row['mean_auroc']} std={row['std_auroc']} "
            f"range=[{row['min_auroc']}, {row['max_auroc']}]"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--detail-output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
