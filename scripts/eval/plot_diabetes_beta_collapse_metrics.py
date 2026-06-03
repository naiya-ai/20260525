"""Plot diabetes beta sweep collapse and non-collapsed evaluation metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-summary", type=Path, required=True)
    parser.add_argument("--coverage-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-group", default="diabetes")
    parser.add_argument("--min-kl-mean", type=float, default=0.05)
    parser.add_argument("--min-active-units", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def beta_label(value: float) -> str:
    if value < 0.01:
        return f"{value:.3f}"
    if value < 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.1f}"


def add_mean_minmax(ax, x_positions, data: pd.DataFrame, column: str, *, color: str) -> None:
    for x, (_, group) in zip(x_positions, data.groupby("beta", sort=True)):
        values = group[column].dropna()
        if values.empty:
            continue
        mean = values.mean()
        lower = mean - values.min()
        upper = values.max() - mean
        ax.errorbar(
            [x],
            [mean],
            yerr=[[lower], [upper]],
            fmt="o",
            markersize=6,
            capsize=5,
            color=color,
            ecolor=color,
            linewidth=1.5,
            zorder=4,
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = pd.read_csv(args.diagnostics_summary)
    coverage = pd.read_csv(args.coverage_summary)

    diag = diagnostics[diagnostics["target_group"].eq(args.target_group)].copy()
    diag["beta"] = diag["beta"].astype(float)
    active_col = "test_active_units_kl_gt_0_001"
    if active_col not in diag.columns:
        active_col = "test_active_units_mu_var_gt_0_01"
    diag["noncollapsed"] = (
        diag["test_kl_mean"].ge(args.min_kl_mean)
        & diag[active_col].fillna(0).ge(args.min_active_units)
    )

    cov = coverage[
        coverage["target_group"].eq(args.target_group)
        & coverage["interval_level"].astype(float).round(2).eq(0.90)
    ].copy()
    cov["beta"] = cov["beta"].astype(float)
    cov_run = (
        cov.groupby(["beta", "rep"], as_index=False)
        .agg(
            cov90_abs_error=("abs_coverage_error", "mean"),
            cov90_width=("mean_interval_width", "mean"),
        )
    )

    runs = diag.merge(cov_run, on=["beta", "rep"], how="left")
    betas = sorted(runs["beta"].unique())
    x_positions = list(range(len(betas)))
    beta_to_x = {beta: idx for idx, beta in enumerate(betas)}
    runs["x"] = runs["beta"].map(beta_to_x)
    noncollapsed = runs[runs["noncollapsed"]].copy()

    collapse_summary = (
        runs.groupby("beta", as_index=False)
        .agg(n_total=("rep", "size"), n_noncollapsed=("noncollapsed", "sum"))
    )
    collapse_summary["n_collapsed"] = collapse_summary["n_total"] - collapse_summary["n_noncollapsed"]
    collapse_summary["collapse_rate"] = collapse_summary["n_collapsed"] / collapse_summary["n_total"]

    metric_summary = (
        noncollapsed.groupby("beta", as_index=False)
        .agg(
            n_noncollapsed=("rep", "size"),
            coverage90_abs_error_mean=("cov90_abs_error", "mean"),
            coverage90_abs_error_min=("cov90_abs_error", "min"),
            coverage90_abs_error_max=("cov90_abs_error", "max"),
            prior_rc_mae_mean=("test_prior_transformed_mae", "mean"),
            prior_rc_mae_min=("test_prior_transformed_mae", "min"),
            prior_rc_mae_max=("test_prior_transformed_mae", "max"),
            auroc_mean=("test_auroc", "mean"),
            auroc_min=("test_auroc", "min"),
            auroc_max=("test_auroc", "max"),
        )
    )
    out_csv = args.output_dir / f"{args.target_group}_beta_collapse_noncollapsed_metrics.csv"
    collapse_summary.merge(metric_summary, on="beta", how="left").to_csv(out_csv, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13.33, 7.5), dpi=args.dpi)
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.set_facecolor("white")
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([beta_label(beta) for beta in betas])
        ax.set_xlabel("beta")

    ax = axes[0, 0]
    ax.bar(x_positions, collapse_summary["collapse_rate"], color="#9CA3AF", width=0.58)
    for x, row in zip(x_positions, collapse_summary.itertuples(index=False)):
        ax.text(x, row.collapse_rate + 0.035, f"{int(row.n_collapsed)}/{int(row.n_total)}", ha="center", fontsize=9)
    ax.set_title("Collapse probability", fontweight="bold")
    ax.set_ylabel("collapsed / total")
    ax.set_ylim(0, 1.12)

    panels = [
        (axes[0, 1], "cov90_abs_error", "Coverage calibration", "|coverage90 - 0.90|", "#2563EB"),
        (axes[1, 0], "test_prior_transformed_mae", "Prior reconstruction", "transformed MAE", "#7C3AED"),
        (axes[1, 1], "test_auroc", "AUROC", "test AUROC", "#EA580C"),
    ]
    for ax, column, title, ylabel, color in panels:
        for beta, group in noncollapsed.groupby("beta", sort=True):
            x = beta_to_x[beta]
            offsets = [0] if len(group) == 1 else [(-0.16 + 0.32 * i / max(len(group) - 1, 1)) for i in range(len(group))]
            ax.scatter(
                [x + offset for offset in offsets],
                group[column],
                s=32,
                color=color,
                alpha=0.58,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        add_mean_minmax(ax, x_positions, noncollapsed, column, color=color)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        for x, beta in enumerate(betas):
            n = int(noncollapsed[noncollapsed["beta"].eq(beta)].shape[0])
            ax.text(x, ax.get_ylim()[0], f"n={n}", ha="center", va="bottom", fontsize=8, color="#4B5563")

    axes[0, 1].axhline(0.0, color="#6B7280", linewidth=0.8)
    axes[1, 1].set_ylim(0.45, 0.90)

    fig.suptitle(
        "Diabetes CVAE beta sweep: collapse and non-collapsed evaluation",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.02,
        0.02,
        f"Non-collapsed: KL mean >= {args.min_kl_mean:g} and active units >= {args.min_active_units}. "
        "Dots are runs; marker with error bar is mean with min-max span.",
        fontsize=9,
        color="#6B7280",
    )
    fig.tight_layout(rect=[0.02, 0.05, 0.99, 0.94])

    png = args.output_dir / f"{args.target_group}_beta_collapse_noncollapsed_metrics.png"
    svg = args.output_dir / f"{args.target_group}_beta_collapse_noncollapsed_metrics.svg"
    fig.savefig(png, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)

    print(f"wrote {out_csv}")
    print(f"wrote {png}")
    print(f"wrote {svg}")


if __name__ == "__main__":
    main()
