"""Plot raw harmonized UMAP colored by continuous year, hiding year/survey from features."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EXCLUDE_COLUMNS = {
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
EXCLUDE_PREFIXES = ("wt_",)
YEAR_BIN_EDGES = (1988, 1998, 2005, 2013, 2019, 2025)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harmonized-csv", type=Path, default=Path("datasets/harmonized/harmonized_knhanes_nhanes.csv"))
    parser.add_argument("--max-points", type=int, default=10000)
    parser.add_argument("--sample-mode", choices=["balanced_year_bins", "random"], default="balanced_year_bins")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.05)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/raw_harmonized_umap_year_heatmap"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures/raw_harmonized_umap_by_source"))
    parser.add_argument("--no-missing-indicators", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(
        args.harmonized_csv,
        usecols=lambda col: col in {"source", "source_year", "survey", "year"},
        encoding="utf-8-sig",
    )
    source = meta["source"].astype(str).str.upper()
    year = pd.to_numeric(meta["source_year"], errors="coerce")
    keep = (
        ((source.eq("KNHANES")) & year.ge(1998) & year.le(2024))
        | ((source.eq("NHANES")) & year.ge(1988) & year.le(2023))
    )
    row_indices = np.flatnonzero(keep.to_numpy())
    years = year.to_numpy(dtype=np.float64)[row_indices]
    chosen_local = sample_rows(years, args)
    chosen_rows = row_indices[chosen_local]
    years = years[chosen_local]

    raw = pd.read_csv(args.harmonized_csv, encoding="utf-8-sig")
    feature_columns = select_feature_columns(raw.columns)
    features = raw.iloc[chosen_rows][feature_columns].apply(pd.to_numeric, errors="coerce")
    values = transform_features(features, add_indicators=not args.no_missing_indicators)
    embedding = run_umap(values, args)

    result = pd.DataFrame(
        {
            "row_index": chosen_rows,
            "source_year": years,
            "umap1": embedding[:, 0],
            "umap2": embedding[:, 1],
            "umap3": embedding[:, 2],
        }
    )
    result.to_csv(args.output_dir / "raw_harmonized_umap3d_year_heatmap.csv", index=False)
    result.to_csv(args.figures_dir / "raw_harmonized_umap3d_year_heatmap.csv", index=False)

    pd.DataFrame({"feature": feature_columns}).to_csv(args.output_dir / "raw_umap_features.csv", index=False)
    pd.DataFrame({"feature": feature_columns}).to_csv(args.figures_dir / "raw_umap_features_year_heatmap.csv", index=False)

    plot_pairwise(embedding, years, args.figures_dir / "raw_harmonized_umap_pairwise_year_heatmap.png")
    plot_3d_static(embedding, years, args.figures_dir / "raw_harmonized_umap3d_year_heatmap.png")
    plot_3d_html(embedding, years, args.figures_dir / "raw_harmonized_umap3d_year_heatmap.html")

    print(year_summary(years).to_string(index=False))
    print(f"features={len(feature_columns)} rows={len(years)} transformed_dim={values.shape[1]}")
    print(f"wrote {args.figures_dir / 'raw_harmonized_umap_pairwise_year_heatmap.png'}")
    print(f"wrote {args.figures_dir / 'raw_harmonized_umap3d_year_heatmap.html'}")


def sample_rows(years: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    n = len(years)
    if args.max_points <= 0 or n <= args.max_points:
        return np.arange(n)
    rng = np.random.default_rng(args.seed)
    if args.sample_mode == "random":
        return np.sort(rng.choice(np.arange(n), size=args.max_points, replace=False))
    chosen = []
    per_bin = max(1, args.max_points // (len(YEAR_BIN_EDGES) - 1))
    for start, end in zip(YEAR_BIN_EDGES[:-1], YEAR_BIN_EDGES[1:], strict=True):
        idx = np.flatnonzero((years >= start) & (years < end))
        if len(idx) == 0:
            continue
        chosen.append(rng.choice(idx, size=min(per_bin, len(idx)), replace=False))
    out = np.concatenate(chosen)
    if len(out) > args.max_points:
        out = rng.choice(out, size=args.max_points, replace=False)
    return np.sort(out)


def select_feature_columns(columns: pd.Index) -> list[str]:
    selected = []
    for column in columns:
        if column in EXCLUDE_COLUMNS:
            continue
        if any(column.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
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


def plot_pairwise(embedding: np.ndarray, years: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=180)
    pairs = [(0, 1), (0, 2), (1, 2)]
    scatter = None
    for ax, (x_idx, y_idx) in zip(axes, pairs, strict=True):
        scatter = ax.scatter(
            embedding[:, x_idx],
            embedding[:, y_idx],
            c=years,
            cmap="viridis",
            vmin=1988,
            vmax=2024,
            s=5,
            alpha=0.72,
            linewidths=0,
        )
        ax.set_xlabel(f"UMAP {x_idx + 1}")
        ax.set_ylabel(f"UMAP {y_idx + 1}")
        ax.grid(True, alpha=0.18, linewidth=0.7)
    fig.colorbar(scatter, ax=axes, shrink=0.86, pad=0.02, label="Year")
    fig.suptitle("Raw harmonized rows UMAP colored by continuous year", fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 0.94, 0.94))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_3d_static(embedding: np.ndarray, years: np.ndarray, path: Path) -> None:
    fig = plt.figure(figsize=(10.5, 8.2), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        embedding[:, 2],
        c=years,
        cmap="viridis",
        vmin=1988,
        vmax=2024,
        s=7,
        alpha=0.72,
        depthshade=False,
        linewidths=0,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    ax.set_title("Raw harmonized rows UMAP colored by continuous year", fontweight="bold")
    fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.08, label="Year")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_3d_html(embedding: np.ndarray, years: np.ndarray, path: Path) -> None:
    import plotly.graph_objects as go

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=embedding[:, 0],
                y=embedding[:, 1],
                z=embedding[:, 2],
                mode="markers",
                marker={
                    "size": 2.4,
                    "opacity": 0.7,
                    "color": years,
                    "colorscale": "Viridis",
                    "cmin": 1988,
                    "cmax": 2024,
                    "colorbar": {"title": "Year"},
                },
                text=[f"year={int(y)}" for y in years],
            )
        ]
    )
    fig.update_layout(
        title="Raw harmonized rows UMAP colored by continuous year",
        scene={"xaxis_title": "UMAP 1", "yaxis_title": "UMAP 2", "zaxis_title": "UMAP 3"},
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
    )
    fig.write_html(path)


def year_summary(years: np.ndarray) -> pd.DataFrame:
    rows = []
    for start, end in zip(YEAR_BIN_EDGES[:-1], YEAR_BIN_EDGES[1:], strict=True):
        mask = (years >= start) & (years < end)
        rows.append({"year_bin": f"{start}-{end - 1}", "n": int(mask.sum())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
