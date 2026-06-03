"""Plot UMAP of raw harmonized rows, hiding survey/year columns from features."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SOURCE_CLASS_ORDER = ("KNHANES 2013-2024", "KNHANES 1998-2012", "NHANES 1988-2023")
SOURCE_CLASS_COLORS = {
    "KNHANES 2013-2024": "#2563EB",
    "KNHANES 1998-2012": "#F59E0B",
    "NHANES 1988-2023": "#16A34A",
}
YEAR_BIN_EDGES = (1988, 1998, 2005, 2013, 2019, 2025)
YEAR_CLASS_ORDER = (
    "1988-1997",
    "1998-2004",
    "2005-2012",
    "2013-2018",
    "2019-2024",
)
YEAR_CLASS_COLORS = {
    "1988-1997": "#4B5563",
    "1998-2004": "#7C3AED",
    "2005-2012": "#DC2626",
    "2013-2018": "#F59E0B",
    "2019-2024": "#2563EB",
}
DEFAULT_EXCLUDE_COLUMNS = {
    "source",
    "source_year",
    "source_cycle",
    "survey",
    "year",
    "row_index",
    "mod_d",
    "ID",
    "ID_fam",
    "ID_F",
    "ID_M",
    "psu",
    "kstrata",
}
DEFAULT_EXCLUDE_PREFIXES = ("wt_",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harmonized-csv", type=Path, default=Path("datasets/harmonized/harmonized_knhanes_nhanes.csv"))
    parser.add_argument("--max-points", type=int, default=10000)
    parser.add_argument("--sample-mode", choices=["balanced", "proportional"], default="balanced")
    parser.add_argument("--class-by", choices=["source", "year"], default="source")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.05)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/raw_harmonized_umap_by_source"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures/raw_harmonized_umap_by_source"))
    parser.add_argument("--extra-exclude-columns", nargs="*", default=[])
    parser.add_argument("--keep-weights", action="store_true")
    parser.add_argument("--no-missing-indicators", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    labels_meta = pd.read_csv(
        args.harmonized_csv,
        usecols=lambda col: col in {"source", "source_year", "survey", "year"},
        encoding="utf-8-sig",
    )
    labels, class_order, class_colors = make_labels(labels_meta, args.class_by)
    keep = np.isin(labels, class_order)
    row_indices = np.flatnonzero(keep)
    labels = labels[keep]
    chosen_local = stratified_sample(
        labels,
        max_points=args.max_points,
        seed=args.seed,
        mode=args.sample_mode,
        class_order=class_order,
    )
    chosen_rows = row_indices[chosen_local]
    labels = labels[chosen_local]

    raw = pd.read_csv(args.harmonized_csv, encoding="utf-8-sig")
    feature_columns = select_feature_columns(raw.columns, args)
    features = raw.iloc[chosen_rows][feature_columns].apply(pd.to_numeric, errors="coerce")
    transformed = transform_features(features, add_indicators=not args.no_missing_indicators)
    embedding = run_umap(transformed, args)

    result = pd.DataFrame(
        {
            "row_index": chosen_rows,
            "source_class": labels,
            "umap1": embedding[:, 0],
            "umap2": embedding[:, 1],
            "umap3": embedding[:, 2],
        }
    )
    result.to_csv(args.output_dir / "raw_harmonized_umap3d_by_source.csv", index=False)
    result.to_csv(args.figures_dir / "raw_harmonized_umap3d_by_source.csv", index=False)
    pd.DataFrame({"feature": feature_columns}).to_csv(args.output_dir / "raw_umap_features.csv", index=False)
    pd.DataFrame({"feature": feature_columns}).to_csv(args.figures_dir / "raw_umap_features.csv", index=False)

    suffix = f"by_{args.class_by}"
    plot_pairwise(
        embedding,
        labels,
        args.figures_dir / f"raw_harmonized_umap_pairwise_{suffix}.png",
        class_order,
        class_colors,
        args,
    )
    plot_3d_static(
        embedding,
        labels,
        args.figures_dir / f"raw_harmonized_umap3d_{suffix}.png",
        class_order,
        class_colors,
        args,
    )
    plot_3d_html(
        embedding,
        labels,
        args.figures_dir / f"raw_harmonized_umap3d_{suffix}.html",
        class_order,
        class_colors,
        args,
    )

    print(class_counts(labels, class_order).to_string(index=False))
    print(f"features={len(feature_columns)} rows={len(labels)} transformed_dim={transformed.shape[1]}")
    print(f"wrote {args.figures_dir / f'raw_harmonized_umap_pairwise_{suffix}.png'}")
    print(f"wrote {args.figures_dir / f'raw_harmonized_umap3d_{suffix}.html'}")


def make_labels(meta: pd.DataFrame, class_by: str) -> tuple[np.ndarray, tuple[str, ...], dict[str, str]]:
    if class_by == "year":
        return year_labels(meta), YEAR_CLASS_ORDER, YEAR_CLASS_COLORS
    return source_labels(meta), SOURCE_CLASS_ORDER, SOURCE_CLASS_COLORS


def source_labels(meta: pd.DataFrame) -> np.ndarray:
    source = meta["source"].astype(str).str.upper()
    year = pd.to_numeric(meta["source_year"], errors="coerce")
    labels = np.full(len(meta), "other", dtype=object)
    labels[(source.eq("KNHANES")) & year.ge(2013) & year.le(2024)] = "KNHANES 2013-2024"
    labels[(source.eq("KNHANES")) & year.ge(1998) & year.lt(2013)] = "KNHANES 1998-2012"
    labels[(source.eq("NHANES")) & year.ge(1988) & year.le(2023)] = "NHANES 1988-2023"
    return labels


def year_labels(meta: pd.DataFrame) -> np.ndarray:
    year = pd.to_numeric(meta["source_year"], errors="coerce")
    labels = np.full(len(meta), "other", dtype=object)
    for start, end in zip(YEAR_BIN_EDGES[:-1], YEAR_BIN_EDGES[1:], strict=True):
        labels[year.ge(start) & year.lt(end)] = f"{start}-{end - 1}"
    return labels


def stratified_sample(
    labels: np.ndarray,
    *,
    max_points: int,
    seed: int,
    mode: str,
    class_order: tuple[str, ...],
) -> np.ndarray:
    n = len(labels)
    if max_points <= 0 or n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    chosen = []
    if mode == "balanced":
        per_class = max(1, max_points // len(class_order))
        for label in class_order:
            idx = np.flatnonzero(labels == label)
            chosen.append(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
        return np.sort(np.concatenate(chosen))
    for label in class_order:
        idx = np.flatnonzero(labels == label)
        quota = max(1, int(round(max_points * len(idx) / n)))
        chosen.append(rng.choice(idx, size=min(quota, len(idx)), replace=False))
    out = np.concatenate(chosen)
    if len(out) > max_points:
        out = rng.choice(out, size=max_points, replace=False)
    return np.sort(out)


def select_feature_columns(columns: pd.Index, args: argparse.Namespace) -> list[str]:
    excluded = set(DEFAULT_EXCLUDE_COLUMNS) | set(args.extra_exclude_columns)
    prefixes = () if args.keep_weights else DEFAULT_EXCLUDE_PREFIXES
    selected = []
    for column in columns:
        if column in excluded:
            continue
        if any(column.startswith(prefix) for prefix in prefixes):
            continue
        selected.append(column)
    return selected


def transform_features(features: pd.DataFrame, *, add_indicators: bool) -> np.ndarray:
    pipeline = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=add_indicators),
        StandardScaler(),
    )
    return pipeline.fit_transform(features).astype(np.float32, copy=False)


def run_umap(values: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    import umap

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed,
    )
    return reducer.fit_transform(values).astype(np.float64)


def labels_by_descending_count(labels: np.ndarray, class_order: tuple[str, ...]) -> list[str]:
    return sorted(class_order, key=lambda label: int(np.sum(labels == label)), reverse=True)


def plot_pairwise(
    embedding: np.ndarray,
    labels: np.ndarray,
    path: Path,
    class_order: tuple[str, ...],
    class_colors: dict[str, str],
    args: argparse.Namespace,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=180)
    pairs = [(0, 1), (0, 2), (1, 2)]
    for ax, (x_idx, y_idx) in zip(axes, pairs, strict=True):
        for label in labels_by_descending_count(labels, class_order):
            mask = labels == label
            ax.scatter(
                embedding[mask, x_idx],
                embedding[mask, y_idx],
                s=5,
                alpha=0.24,
                color=class_colors[label],
                label=f"{label} (n={int(mask.sum())})",
                linewidths=0,
            )
        ax.set_xlabel(f"UMAP {x_idx + 1}")
        ax.set_ylabel(f"UMAP {y_idx + 1}")
        ax.grid(True, alpha=0.18, linewidth=0.7)
    handles, label_text = axes[0].get_legend_handles_labels()
    fig.legend(handles, label_text, loc="lower center", ncol=min(5, len(class_order)), frameon=False)
    fig.suptitle(
        f"Raw harmonized rows UMAP by {args.class_by}, survey/year hidden from features",
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_3d_static(
    embedding: np.ndarray,
    labels: np.ndarray,
    path: Path,
    class_order: tuple[str, ...],
    class_colors: dict[str, str],
    args: argparse.Namespace,
) -> None:
    fig = plt.figure(figsize=(10.5, 8.2), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    for label in labels_by_descending_count(labels, class_order):
        mask = labels == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],
            s=7,
            alpha=0.38,
            color=class_colors[label],
            label=f"{label} (n={int(mask.sum())})",
            depthshade=False,
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    ax.set_title(f"Raw harmonized rows UMAP by {args.class_by}, survey/year hidden", fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_3d_html(
    embedding: np.ndarray,
    labels: np.ndarray,
    path: Path,
    class_order: tuple[str, ...],
    class_colors: dict[str, str],
    args: argparse.Namespace,
) -> None:
    import plotly.graph_objects as go

    fig = go.Figure()
    for label in labels_by_descending_count(labels, class_order):
        mask = labels == label
        fig.add_trace(
            go.Scatter3d(
                x=embedding[mask, 0],
                y=embedding[mask, 1],
                z=embedding[mask, 2],
                mode="markers",
                name=f"{label} (n={int(mask.sum())})",
                marker={"size": 2.4, "opacity": 0.34, "color": class_colors[label]},
            )
        )
    fig.update_layout(
        title=f"Raw harmonized rows UMAP by {args.class_by}, survey/year hidden",
        scene={"xaxis_title": "UMAP 1", "yaxis_title": "UMAP 2", "zaxis_title": "UMAP 3"},
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
    )
    fig.write_html(path)


def class_counts(labels: np.ndarray, class_order: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"class": label, "n": int(np.sum(labels == label))} for label in class_order]
    )


if __name__ == "__main__":
    main()
