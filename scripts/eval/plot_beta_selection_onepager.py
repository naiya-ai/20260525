"""Create a one-slide beta-selection summary image for CVAE sweeps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle


BETA_COLORS = {
    0.001: "#DCEBFF",
    0.01: "#E0F4E8",
    0.1: "#FFF1CC",
    1.0: "#F3DDE4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--diagnostics-summary", type=Path, required=True)
    parser.add_argument("--diagnostics-dir", type=Path)
    parser.add_argument("--coverage-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="CVAE beta selection summary")
    parser.add_argument("--min-usable-rate", type=float, default=0.5)
    parser.add_argument("--max-prior-transformed-mae", type=float, default=1.5)
    parser.add_argument("--min-kl-mean", type=float, default=0.05)
    parser.add_argument("--min-active-units", type=int, default=1)
    parser.add_argument("--max-cov90-abs-error", type=float, default=0.08)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def infer_target_group_from_checkpoint(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    path = Path(str(value))
    if path.name == "checkpoint_best.pt":
        return path.parent.name
    return path.name


def beta_label(value: object) -> str:
    beta = float(value)
    if beta < 0.01:
        return f"{beta:.3f}"
    if beta < 1:
        return f"{beta:.2f}".rstrip("0").rstrip(".")
    return f"{beta:.1f}"


def add_target_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "target_group" not in df.columns:
        df["target_group"] = df["checkpoint"].map(infer_target_group_from_checkpoint)
    return df


def add_categorical_prior_accuracy(df: pd.DataFrame, diagnostics_dir: Path) -> pd.DataFrame:
    df = df.copy()
    if "test_prior_cat_accuracy" not in df.columns:
        df["test_prior_cat_accuracy"] = pd.NA
    missing = df["test_prior_cat_accuracy"].isna()
    if not missing.any():
        return df

    for index, row in df[missing].iterrows():
        beta_tag = f"{int(float(row['beta']) * 1000 + 0.5):03d}"
        rep = int(row["rep"])
        target_group = str(row["target_group"])
        result_path = diagnostics_dir / f"beta{beta_tag}_rep{rep}_{target_group}" / (
            f"beta{beta_tag}_rep{rep}_{target_group}_diagnostics.json"
        )
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        recon = result.get("splits", {}).get("test", {}).get("conditional_prior_reconstruction", {})
        df.at[index, "test_prior_cat_accuracy"] = recon.get("cat_accuracy")
    return df


def aggregate_coverage(coverage: pd.DataFrame) -> pd.DataFrame:
    coverage = coverage.copy()
    coverage = coverage[coverage["interval_level"].astype(float).round(2) == 0.90]
    keys = ["target_group", "beta", "rep"]
    fields = ["abs_coverage_error", "mean_interval_width", "abs_mean_bias"]
    existing = [field for field in fields if field in coverage.columns]
    return (
        coverage.groupby(keys, as_index=False)[existing]
        .mean(numeric_only=True)
        .rename(
            columns={
                "abs_coverage_error": "cov90_abs_error",
                "mean_interval_width": "cov90_width",
                "abs_mean_bias": "cov90_abs_bias",
            }
        )
    )


def load_total_counts(status_file: Path | None, diagnostics: pd.DataFrame) -> pd.DataFrame:
    if status_file is None or not status_file.exists():
        return (
            diagnostics.groupby(["target_group", "beta"], as_index=False)
            .size()
            .rename(columns={"size": "n_total"})
        )
    status = pd.read_csv(status_file, sep="\t")
    status = status[status["exit_code"].astype(str) == "0"].copy()
    status["target_group"] = status["run_dir"].map(lambda value: Path(str(value)).name)
    status["beta"] = status["beta"].astype(float)
    return (
        status.groupby(["target_group", "beta"], as_index=False)
        .size()
        .rename(columns={"size": "n_total"})
    )


def mark_usable(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = df.copy()
    active_col = "test_active_units_kl_gt_0_001"
    if active_col not in df.columns:
        active_col = "test_active_units_mu_var_gt_0_01"
    prior_available = df["test_prior_transformed_mae"].notna() | df["test_prior_cat_accuracy"].notna()
    prior_ok = df["test_prior_transformed_mae"].le(args.max_prior_transformed_mae) | df["test_prior_cat_accuracy"].notna()
    df["usable"] = (
        df["test_auroc"].notna()
        & prior_available
        & prior_ok
        & df["test_kl_mean"].ge(args.min_kl_mean)
        & df[active_col].fillna(0).ge(args.min_active_units)
    )
    # Categorical-only targets can still be useful for ranking/classification even when
    # the latent KL criterion says "collapsed"; keep their AUROC/prior-RC comparison
    # visible and mark the KL status separately in the report.
    categorical_only = df["test_prior_transformed_mae"].isna() & df["test_prior_cat_accuracy"].notna()
    df.loc[categorical_only, "usable"] = df.loc[categorical_only, "test_auroc"].notna()
    df["kl_alive"] = df["test_kl_mean"].ge(args.min_kl_mean) & df[active_col].fillna(0).ge(args.min_active_units)
    return df


def normalize_score(values: pd.Series, higher_is_better: bool) -> pd.Series:
    values = values.astype(float)
    span = values.max() - values.min()
    if not math.isfinite(span) or span == 0:
        return pd.Series(0.5, index=values.index)
    if higher_is_better:
        return (values - values.min()) / span
    return (values.max() - values) / span


def aggregate_metrics(
    diagnostics: pd.DataFrame,
    coverage: pd.DataFrame,
    total_counts: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = diagnostics.merge(coverage, on=["target_group", "beta", "rep"], how="left")
    merged = mark_usable(merged, args)
    usable = merged[merged["usable"]].copy()

    grouped = (
        usable.groupby(["target_group", "beta"], as_index=False)
        .agg(
            n_usable=("usable", "size"),
            n_kl_alive=("kl_alive", "sum"),
            auroc_mean=("test_auroc", "mean"),
            auroc_max=("test_auroc", "max"),
            prior_transformed_mae_mean=("test_prior_transformed_mae", "mean"),
            prior_cat_accuracy_mean=("test_prior_cat_accuracy", "mean"),
            cov90_abs_error_mean=("cov90_abs_error", "mean"),
            cov90_width_mean=("cov90_width", "mean"),
        )
    )
    metrics = total_counts.merge(grouped, on=["target_group", "beta"], how="left")
    metrics["n_usable"] = metrics["n_usable"].fillna(0).astype(int)
    metrics["n_kl_alive"] = metrics["n_kl_alive"].fillna(0).astype(int)
    metrics["usable_rate"] = metrics["n_usable"] / metrics["n_total"].clip(lower=1)
    metrics["kl_alive_rate"] = metrics["n_kl_alive"] / metrics["n_total"].clip(lower=1)
    metrics["n_collapsed"] = metrics["n_total"] - metrics["n_usable"]
    metrics["prior_rc_error_mean"] = metrics["prior_transformed_mae_mean"]
    cat_error = 1.0 - metrics["prior_cat_accuracy_mean"]
    metrics["prior_rc_error_mean"] = metrics["prior_rc_error_mean"].combine_first(cat_error)
    metrics["prior_rc_label"] = "num MAE"
    metrics.loc[metrics["prior_transformed_mae_mean"].isna() & metrics["prior_cat_accuracy_mean"].notna(), "prior_rc_label"] = "cat error"

    selections = []
    for target_group, group in metrics.groupby("target_group", sort=True):
        group = group.copy()
        eligible = group[
            group["usable_rate"].ge(args.min_usable_rate)
            & (
                group["cov90_abs_error_mean"].fillna(0.0).le(args.max_cov90_abs_error)
                | group["cov90_abs_error_mean"].isna()
            )
            & group["auroc_mean"].notna()
            & group["prior_rc_error_mean"].notna()
        ].copy()
        if eligible.empty:
            eligible = group[group["n_usable"].gt(0)].copy()
            fallback = True
        else:
            fallback = False
        if eligible.empty:
            eligible = group.copy()
            fallback = True

        eligible["score"] = (
            0.35 * normalize_score(eligible["auroc_mean"].fillna(0), True)
            + 0.25 * normalize_score(eligible["prior_rc_error_mean"].fillna(999), False)
            + 0.25 * normalize_score(eligible["cov90_abs_error_mean"].fillna(999), False)
            + 0.15 * normalize_score(eligible["usable_rate"].fillna(0), True)
        )
        selected = eligible.sort_values(["score", "usable_rate", "auroc_mean"], ascending=False).iloc[0].copy()
        selected["target_group"] = target_group
        selected["fallback"] = fallback
        selections.append(selected)

    selected_df = pd.DataFrame(selections)
    selected_df["rationale"] = selected_df.apply(make_rationale, axis=1)
    return metrics.sort_values(["target_group", "beta"]), selected_df.sort_values("target_group")


def make_rationale(row: pd.Series) -> str:
    if bool(row.get("fallback", False)):
        return "fallback: check collapse/calibration"
    usable_rate = float(row.get("usable_rate", 0))
    cov_error = row.get("cov90_abs_error_mean")
    prior_mae = row.get("prior_transformed_mae_mean")
    if pd.isna(row.get("cov90_abs_error_mean")) and pd.notna(row.get("prior_cat_accuracy_mean")):
        return "categorical target"
    if usable_rate >= 0.8 and pd.notna(cov_error) and cov_error <= 0.03:
        return "stable + calibrated"
    if pd.notna(prior_mae) and prior_mae <= 0.75:
        return "good prior RC"
    if pd.notna(cov_error) and cov_error <= 0.04:
        return "calibration-led"
    return "balanced tradeoff"


def fmt(value: object, digits: int = 3, missing: str = "-") -> str:
    if value is None:
        return missing
    try:
        if pd.isna(value):
            return missing
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def metric_text(row: pd.Series, metric: str) -> tuple[str, str]:
    if metric == "usable":
        return f"{int(row['n_usable'])}/{int(row['n_total'])}", "higher"
    if metric == "auroc":
        return fmt(row.get("auroc_mean"), 3), "higher"
    if metric == "coverage":
        return fmt(row.get("cov90_abs_error_mean"), 3, "N/A"), "lower"
    if metric == "prior":
        label = "CE" if row.get("prior_rc_label") == "cat error" else "MAE"
        value = fmt(row.get("prior_rc_error_mean"), 3, "N/A")
        return f"{value} {label}" if value != "N/A" else value, "lower"
    raise ValueError(metric)


def metric_intensity(group: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "usable":
        return group["usable_rate"].fillna(0)
    if metric == "auroc":
        return normalize_score(group["auroc_mean"].fillna(0), True)
    if metric == "coverage":
        values = group["cov90_abs_error_mean"]
        if values.notna().sum() == 0:
            return pd.Series(0.15, index=group.index)
        return normalize_score(values.fillna(values.max()), False)
    if metric == "prior":
        values = group["prior_rc_error_mean"]
        if values.notna().sum() == 0:
            return pd.Series(0.15, index=group.index)
        return normalize_score(values.fillna(values.max()), False)
    raise ValueError(metric)


def blend_with_white(hex_color: str, amount: float) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    base = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    amount = max(0.0, min(1.0, float(amount)))
    return tuple(1.0 - amount * (1.0 - channel) for channel in base)


def draw_comparison_report(metrics: pd.DataFrame, selected: pd.DataFrame, args: argparse.Namespace, output_png: Path, output_svg: Path) -> None:
    diseases = list(selected["target_group"])
    betas = sorted(metrics["beta"].dropna().unique())
    metric_specs = [
        ("usable", "usable"),
        ("auroc", "AUROC"),
        ("coverage", "cov90 |err|"),
        ("prior", "prior RC"),
    ]
    selected_lookup = {row["target_group"]: float(row["beta"]) for _, row in selected.iterrows()}

    fig = plt.figure(figsize=(13.33, 7.5), dpi=args.dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    ax.text(0.04, 0.94, "CVAE beta selection: disease-wise comparison", fontsize=23, fontweight="bold", color="#18202A", va="top")
    ax.text(
        0.04,
        0.895,
        "Each disease compares beta candidates by usable runs, AUROC, coverage calibration, and prior reconstruction. Darker cells are better within disease/metric.",
        fontsize=10.2,
        color="#4B5563",
        va="top",
    )

    left = 0.04
    right = 0.965
    top = 0.835
    disease_w = 0.145
    metric_w = 0.09
    table_w = right - left
    beta_w = (table_w - disease_w - metric_w) / len(betas)
    header_h = 0.052
    row_h = 0.0205

    ax.add_patch(Rectangle((left, top - header_h), table_w, header_h, facecolor="#F3F4F6", edgecolor="none"))
    ax.text(left + 0.006, top - header_h / 2, "Disease", fontsize=9.8, fontweight="bold", color="#374151", va="center")
    ax.text(left + disease_w + 0.006, top - header_h / 2, "Metric", fontsize=9.8, fontweight="bold", color="#374151", va="center")
    for beta_idx, beta in enumerate(betas):
        x = left + disease_w + metric_w + beta_idx * beta_w
        ax.text(x + beta_w / 2, top - header_h / 2, f"beta {beta_label(beta)}", fontsize=9.6, fontweight="bold", color="#374151", va="center", ha="center")

    y = top - header_h
    selected_rows = selected.set_index("target_group")
    for disease_idx, disease in enumerate(diseases):
        disease_metrics = metrics[metrics["target_group"].eq(disease)].copy()
        if disease_metrics.empty:
            continue
        block_h = row_h * len(metric_specs)
        block_bg = "#FFFFFF" if disease_idx % 2 == 0 else "#FAFAFA"
        ax.add_patch(Rectangle((left, y - block_h), table_w, block_h, facecolor=block_bg, edgecolor="#E5E7EB", linewidth=0.5))
        selected_beta = selected_lookup.get(disease)
        selected_row = selected_rows.loc[disease]
        selected_reason = str(selected_row.get("rationale", ""))
        ax.text(left + 0.006, y - block_h / 2 + 0.011, disease, fontsize=9.2, fontweight="bold", color="#111827", va="center")
        ax.text(left + 0.006, y - block_h / 2 - 0.011, f"select {beta_label(selected_beta)}", fontsize=8.2, color="#4B5563", va="center")
        ax.text(left + 0.006, y - block_h + 0.005, selected_reason, fontsize=7.2, color="#6B7280", va="bottom")

        for metric_idx, (metric_key, metric_name) in enumerate(metric_specs):
            row_y_top = y - metric_idx * row_h
            ax.text(left + disease_w + 0.006, row_y_top - row_h / 2, metric_name, fontsize=8.5, color="#374151", va="center")
            intensities = metric_intensity(disease_metrics, metric_key)
            intensity_map = dict(zip(disease_metrics["beta"].astype(float), intensities))
            rows_by_beta = {float(row["beta"]): row for _, row in disease_metrics.iterrows()}
            for beta_idx, beta in enumerate(betas):
                x = left + disease_w + metric_w + beta_idx * beta_w
                row = rows_by_beta.get(float(beta))
                if row is None:
                    text = "-"
                    intensity = 0.0
                else:
                    text, _ = metric_text(row, metric_key)
                    intensity = float(intensity_map.get(float(beta), 0.0))
                color = blend_with_white("#4C78A8", 0.18 + 0.72 * intensity)
                ax.add_patch(Rectangle((x + 0.002, row_y_top - row_h + 0.002), beta_w - 0.004, row_h - 0.004, facecolor=color, edgecolor="none"))
                text_color = "#FFFFFF" if intensity > 0.62 else "#111827"
                ax.text(x + beta_w / 2, row_y_top - row_h / 2, text, fontsize=7.6, color=text_color, va="center", ha="center")
                if selected_beta is not None and abs(float(beta) - float(selected_beta)) < 1e-12:
                    ax.add_patch(Rectangle((x + 0.001, row_y_top - row_h + 0.001), beta_w - 0.002, row_h - 0.002, facecolor="none", edgecolor="#111827", linewidth=1.2))
        y -= block_h

    ax.text(
        0.04,
        0.055,
        "For hepatitis B/C, coverage is N/A because targets are categorical logits; prior RC is categorical error = 1 - prior-sample mean accuracy. Selection is outlined in black.",
        fontsize=8.4,
        color="#6B7280",
        va="bottom",
    )
    fig.savefig(output_png, bbox_inches="tight", facecolor="white")
    fig.savefig(output_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_report(selected: pd.DataFrame, args: argparse.Namespace, output_png: Path, output_svg: Path) -> None:
    fig = plt.figure(figsize=(13.33, 7.5), dpi=args.dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    ax.text(0.045, 0.935, args.title, fontsize=24, fontweight="bold", color="#18202A", va="top")
    subtitle = (
        "Selection rule: usable runs >= 50%, coverage90 not degraded, then prioritize "
        "AUROC, prior reconstruction, and calibration."
    )
    ax.text(0.045, 0.892, subtitle, fontsize=10.5, color="#4B5563", va="top")

    columns = [
        ("Disease", 0.18),
        ("Selected beta", 0.12),
        ("Usable", 0.10),
        ("AUROC", 0.11),
        ("Cov90 |err|", 0.13),
        ("Prior RC MAE", 0.14),
        ("Readout", 0.22),
    ]
    left = 0.045
    right = 0.955
    table_top = 0.835
    row_h = 0.072
    header_h = 0.052
    table_w = right - left
    xs = [left]
    for _, width in columns[:-1]:
        xs.append(xs[-1] + width * table_w)

    ax.add_patch(Rectangle((left, table_top - header_h), table_w, header_h, facecolor="#F3F4F6", edgecolor="none"))
    x = left
    for title, width in columns:
        ax.text(x + 0.008, table_top - header_h / 2, title, fontsize=10, fontweight="bold", color="#374151", va="center")
        x += width * table_w

    beta_counts = selected["beta"].astype(float).value_counts().sort_index()
    for row_idx, (_, row) in enumerate(selected.iterrows()):
        y_top = table_top - header_h - row_idx * row_h
        bg = "#FFFFFF" if row_idx % 2 == 0 else "#FAFAFA"
        ax.add_patch(Rectangle((left, y_top - row_h), table_w, row_h, facecolor=bg, edgecolor="#E5E7EB", linewidth=0.6))

        beta = float(row["beta"])
        beta_bg = BETA_COLORS.get(beta, "#E5E7EB")
        values = [
            str(row["target_group"]),
            beta_label(beta),
            f"{int(row['n_usable'])}/{int(row['n_total'])}",
            fmt(row.get("auroc_mean"), 3),
            fmt(row.get("cov90_abs_error_mean"), 3),
            fmt(row.get("prior_transformed_mae_mean"), 3),
            str(row.get("rationale", "")),
        ]
        x = left
        for col_idx, ((_, width), value) in enumerate(zip(columns, values)):
            cell_w = width * table_w
            if col_idx == 1:
                ax.add_patch(
                    Rectangle(
                        (x + 0.006, y_top - row_h + 0.014),
                        min(cell_w - 0.012, 0.086),
                        row_h - 0.028,
                        facecolor=beta_bg,
                        edgecolor="#CBD5E1",
                        linewidth=0.8,
                    )
                )
                ax.text(x + 0.049, y_top - row_h / 2, value, fontsize=10.5, color="#111827", va="center", ha="center")
            else:
                color = "#111827" if col_idx in (0, 6) else "#253041"
                weight = "bold" if col_idx == 0 else "normal"
                ax.text(x + 0.008, y_top - row_h / 2, value, fontsize=10.2, color=color, va="center", fontweight=weight)
            x += cell_w

    legend_y = 0.118
    ax.text(0.045, legend_y + 0.043, "Selected beta count", fontsize=10.5, fontweight="bold", color="#374151")
    legend_x = 0.045
    for beta, count in beta_counts.items():
        label = f"beta {beta_label(beta)}: {int(count)}"
        color = BETA_COLORS.get(float(beta), "#E5E7EB")
        ax.add_patch(Rectangle((legend_x, legend_y), 0.018, 0.018, facecolor=color, edgecolor="#CBD5E1", linewidth=0.7))
        ax.text(legend_x + 0.024, legend_y + 0.009, label, fontsize=9.5, color="#374151", va="center")
        legend_x += 0.145

    footnote = (
        f"Collapse/usable inferred as AUROC present, prior transformed MAE <= {args.max_prior_transformed_mae:g}, "
        f"KL >= {args.min_kl_mean:g}, active units >= {args.min_active_units}. "
        "Tune thresholds if manual collapse review differs."
    )
    ax.text(0.045, 0.055, footnote, fontsize=8.5, color="#6B7280", va="bottom")

    fig.savefig(output_png, bbox_inches="tight", facecolor="white")
    fig.savefig(output_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = args.diagnostics_dir or args.diagnostics_summary.parent

    diagnostics = add_target_group(pd.read_csv(args.diagnostics_summary))
    diagnostics["beta"] = diagnostics["beta"].astype(float)
    diagnostics = add_categorical_prior_accuracy(diagnostics, diagnostics_dir)
    coverage = pd.read_csv(args.coverage_summary)
    coverage["beta"] = coverage["beta"].astype(float)

    coverage_run = aggregate_coverage(coverage)
    total_counts = load_total_counts(args.status_file, diagnostics)
    metrics, selected = aggregate_metrics(diagnostics, coverage_run, total_counts, args)

    metrics_path = args.output_dir / "beta_selection_metrics.csv"
    selected_path = args.output_dir / "selected_betas.csv"
    png_path = args.output_dir / "beta_selection_onepager.png"
    svg_path = args.output_dir / "beta_selection_onepager.svg"
    comparison_png_path = args.output_dir / "beta_selection_comparison_onepager.png"
    comparison_svg_path = args.output_dir / "beta_selection_comparison_onepager.svg"

    metrics.to_csv(metrics_path, index=False)
    selected.to_csv(selected_path, index=False)
    draw_report(selected, args, png_path, svg_path)
    draw_comparison_report(metrics, selected, args, comparison_png_path, comparison_svg_path)

    print(f"wrote {metrics_path}")
    print(f"wrote {selected_path}")
    print(f"wrote {png_path}")
    print(f"wrote {svg_path}")
    print(f"wrote {comparison_png_path}")
    print(f"wrote {comparison_svg_path}")


if __name__ == "__main__":
    main()
