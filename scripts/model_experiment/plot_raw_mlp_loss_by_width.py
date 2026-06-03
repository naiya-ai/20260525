#!/usr/bin/env python
"""Plot train/valid loss curves across raw MLP classifier widths."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    args = parse_args()
    runs = collect_runs(args.root)
    if not runs:
        print(f"No metrics.csv files found under {args.root}")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = plot_runs(runs, args.output_dir, args.prefix, smooth_window_steps=0)
    if args.smooth_window_steps > 0:
        plot_runs(
            runs,
            args.output_dir,
            f"{args.prefix}_smooth{args.smooth_window_steps}",
            smooth_window_steps=args.smooth_window_steps,
        )
    summary_path = args.output_dir / f"{args.prefix}_last_metrics.csv"
    summary.to_csv(summary_path, index=False)
    print(args.output_dir / f"{args.prefix}.png")
    print(args.output_dir / f"{args.prefix}_zoom.png")
    print(args.output_dir / f"{args.prefix}_tight.png")
    print(summary_path)
    print(summary.to_string(index=False))


def collect_runs(root: Path) -> list[tuple[int, Path, pd.DataFrame]]:
    runs = []
    for metrics_path in root.glob("models/mlp/width_*/rep_01/metrics.csv"):
        match = re.search(r"width_(\d+)", str(metrics_path))
        if match is None:
            continue
        width = int(match.group(1))
        df = pd.read_csv(metrics_path)
        if df.empty or "loss" not in df.columns:
            continue
        df["loss_num"] = pd.to_numeric(df["loss"], errors="coerce")
        df["step_num"] = pd.to_numeric(df["step"], errors="coerce")
        df = df[df["loss_num"].notna() & df["step_num"].notna()].copy()
        if df.empty:
            continue
        runs.append((width, metrics_path, df))
    return sorted(runs, key=lambda item: item[0])


def plot_runs(
    runs: list[tuple[int, Path, pd.DataFrame]],
    output_dir: Path,
    prefix: str,
    *,
    smooth_window_steps: int,
) -> pd.DataFrame:
    rows = []
    for ylim, suffix in [(None, ""), ((0.0, 2.0), "_zoom"), ((0.0, 1.0), "_tight")]:
        plt.figure(figsize=(11, 6.5))
        colors = plt.cm.turbo([index / max(1, len(runs) - 1) for index in range(len(runs))])
        for color, (width, _path, df) in zip(colors, runs):
            for split, style, alpha in [("train", "--", 0.55), ("valid", "-", 0.95)]:
                group = df[df["split"] == split]
                if group.empty:
                    continue
                group = smooth_loss(group, smooth_window_steps)
                plt.plot(
                    group["step_num"],
                    group["plot_loss"],
                    linestyle=style,
                    linewidth=1.5,
                    color=color,
                    alpha=alpha,
                    label=f"{width} {split}",
                )
        plt.xlabel("step")
        plt.ylabel("BCE loss")
        plt.title("Train/valid loss by width")
        plt.grid(True, alpha=0.25)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}{suffix}.png", dpi=180)
        plt.close()

    for width, metrics_path, df in runs:
        train = df[df["split"] == "train"]
        valid = df[df["split"] == "valid"]
        last_train = train.iloc[-1].to_dict() if not train.empty else {}
        last_valid = valid.iloc[-1].to_dict() if not valid.empty else {}
        rows.append(
            {
                "width": width,
                "metrics_path": str(metrics_path),
                "last_train_step": int(last_train.get("step_num", -1)) if last_train else "",
                "last_train_loss": float(last_train.get("loss_num", float("nan"))) if last_train else float("nan"),
                "last_valid_step": int(last_valid.get("step_num", -1)) if last_valid else "",
                "last_valid_loss": float(last_valid.get("loss_num", float("nan"))) if last_valid else float("nan"),
                "last_valid_auroc": float(last_valid.get("auroc", float("nan")))
                if last_valid and str(last_valid.get("auroc", "")) != ""
                else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("width")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="train_valid_loss_by_width")
    parser.add_argument("--smooth-window-steps", type=int, default=1000)
    return parser.parse_args()


def smooth_loss(group: pd.DataFrame, window_steps: int) -> pd.DataFrame:
    group = group.sort_values("step_num").copy()
    if window_steps <= 0 or len(group) < 2:
        group["plot_loss"] = group["loss_num"]
        return group
    step_diffs = group["step_num"].diff().dropna()
    median_step = float(step_diffs.median()) if not step_diffs.empty else float(window_steps)
    window_points = max(1, int(round(window_steps / max(median_step, 1.0))))
    group["plot_loss"] = group["loss_num"].rolling(window=window_points, min_periods=1, center=False).mean()
    return group


if __name__ == "__main__":
    main()
