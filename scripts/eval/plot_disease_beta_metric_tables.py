"""Render disease-wise beta collapse/metric summary tables as a slide image."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle


DEFAULT_DISEASES = [
    "diabetes",
    "hypertension",
    "dyslipidemia",
    "liver_disease",
    "hepatitis_b",
    "hepatitis_c",
    "kidney_disease",
    "anemia",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-summary", type=Path, required=True)
    parser.add_argument("--coverage-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="disease_beta_metric_table_with_hepatitis")
    parser.add_argument("--diseases", nargs="+", default=DEFAULT_DISEASES)
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


def disease_label(value: str) -> str:
    return value.replace("_", " ")


def build_summary(args: argparse.Namespace) -> pd.DataFrame:
    diagnostics = pd.read_csv(args.diagnostics_summary)
    coverage = pd.read_csv(args.coverage_summary)

    diag = diagnostics[diagnostics["target_group"].isin(args.diseases)].copy()
    diag["beta"] = diag["beta"].astype(float)
    active_col = "test_active_units_kl_gt_0_001"
    if active_col not in diag.columns:
        active_col = "test_active_units_mu_var_gt_0_01"
    diag["noncollapsed"] = (
        diag["test_kl_mean"].ge(args.min_kl_mean)
        & diag[active_col].fillna(0).ge(args.min_active_units)
    )

    cov = coverage[
        coverage["target_group"].isin(args.diseases)
        & coverage["interval_level"].astype(float).round(2).eq(0.90)
    ].copy()
    cov["beta"] = cov["beta"].astype(float)
    cov_run = (
        cov.groupby(["target_group", "beta", "rep"], as_index=False)
        .agg(cov90_abs_error=("abs_coverage_error", "mean"))
    )
    runs = diag.merge(cov_run, on=["target_group", "beta", "rep"], how="left")

    collapse = (
        runs.groupby(["target_group", "beta"], as_index=False)
        .agg(n_total=("rep", "size"), n_noncollapsed=("noncollapsed", "sum"))
    )
    collapse["n_collapsed"] = collapse["n_total"] - collapse["n_noncollapsed"]

    runs["categorical_target"] = runs["test_prior_transformed_mae"].isna() & runs["test_prior_cat_bce"].notna()
    noncollapsed = runs[runs["noncollapsed"]].copy()
    metrics = (
        noncollapsed.groupby(["target_group", "beta"], as_index=False)
        .agg(
            cov90_err=("cov90_abs_error", "mean"),
            prior_rc_mae=("test_prior_transformed_mae", "mean"),
            prior_cat_bce=("test_prior_cat_bce", "mean"),
            auroc_noncollapsed=("test_auroc", "mean"),
        )
    )
    summary = collapse.merge(metrics, on=["target_group", "beta"], how="left")
    cat_metrics = (
        runs[runs["categorical_target"]]
        .groupby(["target_group", "beta"], as_index=False)
        .agg(
            prior_cat_bce_all=("test_prior_cat_bce", "mean"),
            auroc_categorical_all=("test_auroc", "mean"),
        )
    )
    summary = summary.merge(cat_metrics, on=["target_group", "beta"], how="left")
    summary["prior_rc"] = (
        summary["prior_rc_mae"]
        .combine_first(summary["prior_cat_bce"])
        .combine_first(summary["prior_cat_bce_all"])
    )
    summary["auroc"] = summary["auroc_noncollapsed"].combine_first(summary["auroc_categorical_all"])
    summary["target_group"] = pd.Categorical(summary["target_group"], categories=args.diseases, ordered=True)
    return summary.sort_values(["target_group", "beta"])


def fmt(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.4f}"


def draw_table(summary: pd.DataFrame, args: argparse.Namespace) -> tuple[Path, Path]:
    betas = sorted(summary["beta"].dropna().unique())
    diseases = [disease for disease in args.diseases if disease in set(summary["target_group"].astype(str))]

    fig = plt.figure(figsize=(13.33, 7.5), dpi=args.dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    ax.text(0.04, 0.945, "CVAE beta selection metrics by disease", fontsize=23, fontweight="bold", color="#18202A", va="top")
    has_hepatitis = {"hepatitis_b", "hepatitis_c"} & set(diseases)
    subtitle = "Metrics use non-collapsed runs only."
    if has_hepatitis:
        subtitle = "Numeric metrics use non-collapsed runs only. Hepatitis AUROC follows EVALUATION.md softmax-positive protocol; coverage is N/A."
    ax.text(0.04, 0.900, subtitle, fontsize=10.5, color="#4B5563", va="top")

    rows_by_key = {(str(row.target_group), float(row.beta)): row for row in summary.itertuples(index=False)}
    panel_w = 0.292
    panel_h = 0.184
    gap_x = 0.026
    gap_y = 0.052
    start_x = 0.04
    start_y = 0.835
    columns = [
        ("beta", 0.17),
        ("collapse", 0.19),
        ("cov90 err", 0.21),
        ("prior RC", 0.24),
        ("AUROC", 0.19),
    ]

    for disease_idx, disease in enumerate(diseases):
        row_idx = disease_idx // 3
        col_idx = disease_idx % 3
        left = start_x + col_idx * (panel_w + gap_x)
        top = start_y - row_idx * (panel_h + gap_y)

        ax.text(left, top + 0.019, disease_label(disease), fontsize=12.2, fontweight="bold", color="#111827", va="bottom")
        ax.add_patch(Rectangle((left, top - panel_h), panel_w, panel_h, facecolor="#FFFFFF", edgecolor="#D1D5DB", linewidth=0.9))

        header_h = 0.044
        row_h = (panel_h - header_h) / len(betas)
        ax.add_patch(Rectangle((left, top - header_h), panel_w, header_h, facecolor="#F3F4F6", edgecolor="none"))

        x = left
        for title, frac in columns:
            col_w = panel_w * frac
            ax.text(x + col_w / 2, top - header_h / 2, title, fontsize=8.4, fontweight="bold", color="#374151", ha="center", va="center")
            x += col_w

        for beta_idx, beta in enumerate(betas):
            y_top = top - header_h - beta_idx * row_h
            bg = "#FFFFFF" if beta_idx % 2 == 0 else "#FAFAFA"
            ax.add_patch(Rectangle((left, y_top - row_h), panel_w, row_h, facecolor=bg, edgecolor="#E5E7EB", linewidth=0.5))
            row = rows_by_key.get((disease, float(beta)))
            if row is None:
                values = [beta_label(beta), "-", "-", "-", "-"]
            else:
                values = [
                    beta_label(beta),
                    f"{int(row.n_collapsed)}/{int(row.n_total)}",
                    fmt(row.cov90_err),
                    fmt(row.prior_rc),
                    fmt(row.auroc),
                ]
            x = left
            for (title, frac), value in zip(columns, values):
                col_w = panel_w * frac
                weight = "bold" if title == "beta" else "normal"
                ax.text(x + col_w / 2, y_top - row_h / 2, value, fontsize=8.9, color="#1F2937", ha="center", va="center", fontweight=weight)
                x += col_w

    footnote = (
        f"Non-collapsed: KL mean >= {args.min_kl_mean:g} and active units >= {args.min_active_units}. "
        "cov90 err = mean |coverage90 - 0.90| over numerical targets; prior RC = transformed MAE."
    )
    if has_hepatitis:
        footnote += " Hepatitis prior RC = categorical BCE; hepatitis AUROC is reported even when KL/active-unit collapse diagnostics are 5/5."
    ax.text(0.04, 0.045, footnote, fontsize=8.5, color="#6B7280", va="bottom")

    png = args.output_dir / f"{args.output_prefix}.png"
    svg = args.output_dir / f"{args.output_prefix}.svg"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, svg


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(args)
    csv_path = args.output_dir / f"{args.output_prefix}.csv"
    summary.to_csv(csv_path, index=False)
    png, svg = draw_table(summary, args)
    print(f"wrote {csv_path}")
    print(f"wrote {png}")
    print(f"wrote {svg}")


if __name__ == "__main__":
    main()
