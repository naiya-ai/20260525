"""Compare CatBoost, EDDI CVAE, and MLP-source CVAE AUROC using coverage-best runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DISEASES = ["diabetes", "hypertension", "dyslipidemia", "liver_disease", "kidney_disease", "anemia"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eddi-diagnostics", type=Path, required=True)
    parser.add_argument("--eddi-coverage", type=Path, required=True)
    parser.add_argument("--eddi-dataset-name", default="harmonized_knhanes_1998_2024")
    parser.add_argument("--mlp-diagnostics", type=Path, required=True)
    parser.add_argument("--mlp-coverage", type=Path, required=True)
    parser.add_argument("--mlp-dataset-name", default="harmonized_knhanes_1998_2024_plus_nhanes_1988_2023")
    parser.add_argument("--catboost-summary", type=Path, required=True)
    parser.add_argument("--catboost-dataset-name", default="harmonized_knhanes_1998_2024_plus_nhanes_1988_2023")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--interval-level", type=float, default=0.90)
    parser.add_argument("--min-kl-mean", type=float, default=0.05)
    parser.add_argument("--min-active-units", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="boost_eddi_mlp_coverage_best_auroc")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    eddi = select_coverage_best(
        diagnostics_path=args.eddi_diagnostics,
        coverage_path=args.eddi_coverage,
        variant="CVAE-EDDI",
        dataset_name=args.eddi_dataset_name,
        beta=args.beta,
        interval_level=args.interval_level,
        min_kl_mean=args.min_kl_mean,
        min_active_units=args.min_active_units,
    )
    mlp = select_coverage_best(
        diagnostics_path=args.mlp_diagnostics,
        coverage_path=args.mlp_coverage,
        variant="CVAE-MLP",
        dataset_name=args.mlp_dataset_name,
        beta=args.beta,
        interval_level=args.interval_level,
        min_kl_mean=args.min_kl_mean,
        min_active_units=args.min_active_units,
    )
    boost = load_catboost(args.catboost_summary, args.catboost_dataset_name)

    combined = (
        pd.DataFrame({"target_group": DISEASES})
        .merge(boost, on="target_group", how="left")
        .merge(eddi, on="target_group", how="left")
        .merge(mlp, on="target_group", how="left", suffixes=("_eddi", "_mlp"))
    )
    combined.to_csv(args.output_dir / f"{args.output_prefix}.csv", index=False)

    plot_auroc(combined, args)
    plot_coverage_table(combined, args)
    print(f"wrote {args.output_dir / f'{args.output_prefix}.csv'}")
    print(f"wrote {args.output_dir / f'{args.output_prefix}.png'}")
    print(f"wrote {args.output_dir / f'{args.output_prefix}_coverage_table.png'}")


def select_coverage_best(
    *,
    diagnostics_path: Path,
    coverage_path: Path,
    variant: str,
    dataset_name: str,
    beta: float,
    interval_level: float,
    min_kl_mean: float,
    min_active_units: int,
) -> pd.DataFrame:
    diag = pd.read_csv(diagnostics_path)
    cov = pd.read_csv(coverage_path)
    diag["beta"] = diag["beta"].astype(float)
    cov["beta"] = cov["beta"].astype(float)
    active_col = "test_active_units_kl_gt_0_001"
    if active_col not in diag.columns:
        active_col = "test_active_units_mu_var_gt_0_01"
    diag = diag[
        diag["target_group"].isin(DISEASES)
        & diag["beta"].round(6).eq(round(beta, 6))
        & diag["test_auroc"].notna()
    ].copy()
    diag["noncollapsed"] = (
        diag["test_kl_mean"].ge(min_kl_mean)
        & diag[active_col].fillna(0).ge(min_active_units)
    )
    cov = cov[
        cov["target_group"].isin(DISEASES)
        & cov["beta"].round(6).eq(round(beta, 6))
        & cov["interval_level"].round(6).eq(round(interval_level, 6))
    ].copy()
    cov_agg = (
        cov.groupby(["target_group", "beta", "rep", "seed", "checkpoint"], as_index=False)
        .agg(
            cov90_abs_error=("abs_coverage_error", "mean"),
            cov90_mean_width=("mean_interval_width", "mean"),
            cov90_n_features=("feature", "nunique"),
        )
    )
    merged = diag.merge(
        cov_agg,
        on=["target_group", "beta", "rep", "seed", "checkpoint"],
        how="left",
    )
    usable = merged[merged["noncollapsed"]].copy()
    if usable.empty:
        usable = merged.copy()
        usable["selection_note"] = "fallback_no_noncollapsed"
    else:
        usable["selection_note"] = "noncollapsed_coverage_best"
    usable = usable.sort_values(
        ["target_group", "cov90_abs_error", "test_auroc"],
        ascending=[True, True, False],
    )
    best = usable.groupby("target_group", as_index=False).head(1).copy()
    counts = (
        merged.groupby("target_group", as_index=False)
        .agg(n_total=("rep", "size"), n_noncollapsed=("noncollapsed", "sum"))
    )
    best = best.merge(counts, on="target_group", how="left")
    prefix = "eddi" if variant == "CVAE-EDDI" else "mlp"
    return best[
        [
            "target_group",
            "rep",
            "seed",
            "test_auroc",
            "test_kl_mean",
            "cov90_abs_error",
            "cov90_mean_width",
            "n_total",
            "n_noncollapsed",
            "selection_note",
        ]
    ].rename(
        columns={
            "rep": f"{prefix}_rep",
            "seed": f"{prefix}_seed",
            "test_auroc": f"{prefix}_auroc",
            "test_kl_mean": f"{prefix}_kl_mean",
            "cov90_abs_error": f"{prefix}_cov90_abs_error",
            "cov90_mean_width": f"{prefix}_cov90_mean_width",
            "n_total": f"{prefix}_n_total",
            "n_noncollapsed": f"{prefix}_n_noncollapsed",
            "selection_note": f"{prefix}_selection_note",
        }
    ).assign(**{f"{prefix}_dataset_name": dataset_name})


def load_catboost(path: Path, dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df[df["dataset_name"].eq(dataset_name) & df["target_group"].isin(DISEASES)].copy()
    return out[["target_group", "auroc", "n_evaluable"]].rename(
        columns={"auroc": "boost_auroc", "n_evaluable": "boost_n_evaluable"}
    )


def plot_auroc(df: pd.DataFrame, args: argparse.Namespace) -> None:
    x = np.arange(len(df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=args.dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.bar(x - width, df["boost_auroc"], width=width, label="Weighted CatBoost", color="#4C78A8")
    ax.bar(x, df["eddi_auroc"], width=width, label="CVAE-EDDI coverage-best", color="#F58518")
    ax.bar(x + width, df["mlp_auroc"], width=width, label="CVAE-MLP coverage-best", color="#54A24B")

    for i, row in enumerate(df.itertuples(index=False)):
        for offset, value in [(-width, row.boost_auroc), (0, row.eddi_auroc), (width, row.mlp_auroc)]:
            if pd.notna(value):
                ax.text(i + offset, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        if pd.notna(row.eddi_auroc):
            ax.text(
                i,
                max(0.47, row.eddi_auroc - 0.06),
                f"cov {row.eddi_cov90_abs_error:.3f}\n{int(row.eddi_n_noncollapsed)}/{int(row.eddi_n_total)}",
                ha="center",
                va="top",
                fontsize=7,
                color="#7C2D12",
            )
        if pd.notna(row.mlp_auroc):
            ax.text(
                i + width,
                max(0.47, row.mlp_auroc - 0.06),
                f"cov {row.mlp_cov90_abs_error:.3f}\n{int(row.mlp_n_noncollapsed)}/{int(row.mlp_n_total)}",
                ha="center",
                va="top",
                fontsize=7,
                color="#14532D",
            )

    ax.set_title(
        "AUROC Comparison: CatBoost vs CVAE-EDDI vs CVAE-MLP",
        fontsize=17,
        fontweight="bold",
        pad=16,
    )
    ax.set_ylabel("Test AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace("_", "\n") for name in df["target_group"]], fontsize=10)
    ax.set_ylim(0.45, 0.97)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    ax.text(
        0.01,
        -0.17,
        f"CVAE bars select beta={args.beta:g} run with lowest mean abs cov90 error among non-collapsed runs "
        f"(KL >= {args.min_kl_mean:g}, active units >= {args.min_active_units}). Labels show cov90 error and noncollapsed/total.",
        transform=ax.transAxes,
        fontsize=9,
        color="#6B7280",
    )
    fig.tight_layout(rect=[0.02, 0.08, 0.99, 0.96])
    fig.savefig(args.output_dir / f"{args.output_prefix}.png", facecolor="white")
    fig.savefig(args.output_dir / f"{args.output_prefix}.svg", facecolor="white")
    plt.close(fig)


def plot_coverage_table(df: pd.DataFrame, args: argparse.Namespace) -> None:
    table = df[
        [
            "target_group",
            "eddi_auroc",
            "eddi_cov90_abs_error",
            "eddi_cov90_mean_width",
            "mlp_auroc",
            "mlp_cov90_abs_error",
            "mlp_cov90_mean_width",
        ]
    ].copy()
    for column in table.columns[1:]:
        table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    table["target_group"] = table["target_group"].str.replace("_", " ", regex=False)
    fig, ax = plt.subplots(figsize=(13.33, 4.8), dpi=args.dpi)
    fig.patch.set_facecolor("white")
    ax.axis("off")
    columns = [
        "Disease",
        "EDDI AUROC",
        "EDDI cov90 err",
        "EDDI width",
        "MLP AUROC",
        "MLP cov90 err",
        "MLP width",
    ]
    tbl = ax.table(
        cellText=table.to_numpy(),
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.45)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        if row == 0:
            cell.set_facecolor("#F3F4F6")
            cell.set_text_props(weight="bold", color="#111827")
        else:
            cell.set_facecolor("white")
    ax.set_title("Coverage-Best CVAE Runs Used for AUROC Comparison", fontsize=15, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(args.output_dir / f"{args.output_prefix}_coverage_table.png", facecolor="white")
    fig.savefig(args.output_dir / f"{args.output_prefix}_coverage_table.svg", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
