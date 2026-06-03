"""Plot reconstruction loss curves for EDDI z2 beta sweeps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def main() -> None:
    args = parse_args()
    runs = latest_runs_by_beta(args.root, args.min_beta, args.max_beta)
    if not runs:
        raise ValueError("No matching beta runs with metrics.csv were found.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.cm.viridis
    summary_rows: list[dict[str, Any]] = []

    for idx, (beta, metrics_path) in enumerate(sorted(runs.items())):
        df = pd.read_csv(metrics_path)
        color = cmap(idx / max(len(runs) - 1, 1))
        for split, linestyle, alpha in (("train", "-", 0.35), ("valid", "--", 0.95)):
            part = df[df["split"] == split]
            if part.empty:
                continue
            ax.plot(
                part["step"],
                part["loss_reconstruction"],
                linestyle=linestyle,
                linewidth=1.5 if split == "train" else 2.0,
                color=color,
                alpha=alpha,
                label=f"beta={beta:.3f} {split}",
            )
        valid = df[df["split"] == "valid"]
        train = df[df["split"] == "train"]
        if not valid.empty:
            best = valid.loc[valid["loss_reconstruction"].astype(float).idxmin()]
            summary_rows.append(
                {
                    "beta": f"{beta:.3f}",
                    "best_valid_reconstruction_loss": f"{float(best['loss_reconstruction']):.8f}",
                    "best_valid_step": int(best["step"]),
                    "last_valid_reconstruction_loss": f"{float(valid.iloc[-1]['loss_reconstruction']):.8f}",
                    "last_valid_step": int(valid.iloc[-1]["step"]),
                    "last_train_step": int(train["step"].max()) if not train.empty else "",
                    "metrics": str(metrics_path),
                }
            )

    ax.set_title("Diabetes EDDI z2 reconstruction loss by beta")
    ax.set_xlabel("step")
    ax.set_ylabel("reconstruction loss")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)

    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(summary_path)
    print(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--min-beta", type=float, default=0.003)
    parser.add_argument("--max-beta", type=float, default=0.010)
    parser.add_argument(
        "--output",
        default="outputs/plots/diabetes_eddi_z2_beta003_010_reconstruction_loss.png",
    )
    parser.add_argument(
        "--summary-output",
        default="outputs/plots/diabetes_eddi_z2_beta003_010_reconstruction_loss_summary.csv",
    )
    return parser.parse_args()


def latest_runs_by_beta(root: Path, min_beta: float, max_beta: float) -> dict[float, Path]:
    candidates: dict[float, list[Path]] = {}
    for metrics_path in root.glob("conditional_vae_eddi_z2_beta*/**/harmonized_knhanes_1998_2024/diabetes/metrics.csv"):
        beta = read_beta(metrics_path)
        if beta is None or beta < min_beta - 1e-12 or beta > max_beta + 1e-12:
            continue
        candidates.setdefault(round(beta, 3), []).append(metrics_path)
    return {
        beta: max(paths, key=lambda path: path.stat().st_mtime)
        for beta, paths in candidates.items()
    }


def read_beta(metrics_path: Path) -> float | None:
    # Prefer the run/output directory tag because training scripts can override
    # beta via CLI while still copying the base YAML into the output directory.
    for part in metrics_path.parts:
        match = re.search(r"beta(\d+)", part)
        if match:
            digits = match.group(1)
            denominator = 1000.0 if len(digits) >= 3 else 100.0
            return int(digits) / denominator

    run_dir = metrics_path.parent
    config_paths = sorted(run_dir.glob("*.yaml"))
    for config_path in config_paths:
        try:
            config = yaml.safe_load(config_path.read_text())
            return float(config["train"]["beta"])
        except Exception:
            continue
    return None


if __name__ == "__main__":
    main()
