"""Plot transformer CVAE loss curves with optional per-run y offsets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_RUNS = (
    "small d128=outputs/conditional_vae_transformer_small/transformer_small_d128_l4_lr3e5_beta001_diabetes_nogpu0_3_20260523_125710/harmonized_knhanes_1998_2024/diabetes/metrics.csv",
    "d256=outputs/conditional_vae_transformer_medium/transformer_medium_d256_l6_lr2e5_beta001_diabetes_nogpu0_3_20260523_130548/harmonized_knhanes_1998_2024/diabetes/metrics.csv",
)


def main() -> None:
    args = parse_args()
    runs = [parse_run(value) for value in args.run]
    offsets = parse_offsets(args.offsets, len(runs), args.offset_step)

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    fig.subplots_adjust(top=0.88, hspace=0.25)
    metrics = [
        ("loss", "Total loss"),
        ("loss_reconstruction", "Reconstruction loss"),
        ("loss_kl", "KL loss"),
    ]

    colors = plt.cm.tab10.colors
    for run_index, ((label, metrics_path), offset) in enumerate(zip(runs, offsets)):
        df = pd.read_csv(metrics_path)
        color = colors[run_index % len(colors)]
        for ax, (column, title) in zip(axes, metrics):
            for split, linestyle, marker in (("train", "-", None), ("valid", "--", "o")):
                part = df[df["split"] == split]
                if part.empty:
                    continue
                ax.plot(
                    part["step"],
                    part[column] + offset,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=3,
                    linewidth=1.8,
                    color=color,
                    alpha=0.9 if split == "train" else 0.75,
                    label=f"{label} {split} (+{offset:g})",
                )
            ax.set_title(title, fontsize=12)
            ax.set_ylabel(f"{column} + offset")
            ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("step")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(args.title, fontsize=15, y=0.98)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=2, frameon=False)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", default=list(DEFAULT_RUNS), help="LABEL=metrics.csv")
    parser.add_argument("--offset-step", type=float, default=0.08)
    parser.add_argument("--offsets", nargs="*", type=float)
    parser.add_argument("--output", default="outputs/plots/diabetes_transformer_loss_offset.png")
    parser.add_argument("--title", default="Diabetes Transformer CVAE Loss With Per-Model Offsets")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Run must be LABEL=metrics.csv, got {value!r}")
    label, path = value.split("=", 1)
    metrics_path = Path(path)
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    return label, metrics_path


def parse_offsets(offsets: list[float] | None, n_runs: int, offset_step: float) -> list[float]:
    if offsets:
        if len(offsets) != n_runs:
            raise ValueError(f"Expected {n_runs} offsets, got {len(offsets)}")
        return offsets
    return [index * offset_step for index in range(n_runs)]


if __name__ == "__main__":
    main()
