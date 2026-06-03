"""Plot 3D UMAP of CVAE conditional-prior latents on train rows by survey era."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from eval.cvae_common import build_model, load_checkpoint_config_and_schema, resolve_device, set_seed
from train.dataset import create_grouped_cvae_dataloader, load_grouped_cvae_schema


CLASS_ORDER = ("KNHANES 2013-2024", "KNHANES 1998-2012", "NHANES 1988-2023")
CLASS_COLORS = {
    "KNHANES 2013-2024": "#2563EB",
    "KNHANES 1998-2012": "#F59E0B",
    "NHANES 1988-2023": "#16A34A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--checkpoint", type=Path)
    selection.add_argument("--sweep-root", type=Path)
    parser.add_argument("--checkpoint-name", default="checkpoint_latest.pt")
    parser.add_argument("--target-group", default="dyslipidemia")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--dataset-root", default="datasets/preprocessed/gaussian_quantile")
    parser.add_argument("--dataset-name", default="harmonized_knhanes_1998_2024_plus_nhanes_1988_2023")
    parser.add_argument("--harmonized-csv", type=Path, default=Path("datasets/harmonized/harmonized_knhanes_nhanes.csv"))
    parser.add_argument("--split", choices=["train", "valid", "test"], default="train")
    parser.add_argument("--max-points", type=int, default=30000)
    parser.add_argument(
        "--sample-mode",
        choices=["proportional", "balanced"],
        default="proportional",
        help="How to subsample source classes before UMAP.",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-kl-per-instance", type=float, default=0.1)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.05)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figures-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    checkpoint, selected_row = resolve_checkpoint(args)
    output_dir = args.output_dir or default_output_dir(args)
    figures_dir = args.figures_dir or default_figures_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    latents, row_indices = extract_prior_mu(args, checkpoint)
    labels = source_labels(args, row_indices)
    keep = finite_label_mask(labels)
    latents = latents[keep]
    row_indices = row_indices[keep]
    labels = labels[keep]
    chosen = stratified_sample(labels, max_points=args.max_points, seed=args.seed, mode=args.sample_mode)
    latents = latents[chosen]
    row_indices = row_indices[chosen]
    labels = labels[chosen]

    embedding = run_umap(latents, args)
    rows = pd.DataFrame(
        {
            "row_index": row_indices,
            "source_class": labels,
            "umap1": embedding[:, 0],
            "umap2": embedding[:, 1],
            "umap3": embedding[:, 2],
        }
    )
    for idx in range(latents.shape[1]):
        rows[f"prior_mu_{idx}"] = latents[:, idx]
    rows.to_csv(output_dir / "train_prior_mu_umap3d_by_source.csv", index=False)
    rows.to_csv(figures_dir / "train_prior_mu_umap3d_by_source.csv", index=False)

    if selected_row is not None:
        pd.DataFrame([selected_row]).to_csv(output_dir / "selected_checkpoint.csv", index=False)
        pd.DataFrame([selected_row]).to_csv(figures_dir / "selected_checkpoint.csv", index=False)

    plot_matplotlib(embedding, labels, figures_dir / "train_prior_mu_umap3d_by_source.png", args)
    plot_pairwise_projections(embedding, labels, figures_dir / "train_prior_mu_umap_pairwise_by_source.png", args)
    plot_plotly(embedding, labels, figures_dir / "train_prior_mu_umap3d_by_source.html", args)
    print(f"checkpoint={checkpoint}")
    print(class_counts(labels).to_string(index=False))
    print(f"wrote {figures_dir / 'train_prior_mu_umap3d_by_source.png'}")
    print(f"wrote {figures_dir / 'train_prior_mu_umap_pairwise_by_source.png'}")
    print(f"wrote {figures_dir / 'train_prior_mu_umap3d_by_source.html'}")


def resolve_checkpoint(args: argparse.Namespace) -> tuple[Path, dict[str, Any] | None]:
    if args.checkpoint is not None:
        return args.checkpoint, None
    status_path = args.sweep_root / "train_status.tsv"
    if not status_path.exists():
        raise FileNotFoundError(status_path)
    status = pd.read_csv(status_path, sep="\t")
    status = status[status["exit_code"].astype(str).eq("0")].copy()
    status["beta"] = status["beta"].astype(float)
    status["target_group"] = status["run_dir"].map(lambda value: Path(str(value)).name)
    status = status[
        status["target_group"].eq(args.target_group)
        & status["beta"].round(6).eq(round(float(args.beta), 6))
    ].copy()
    if status.empty:
        raise ValueError(f"No completed runs for target_group={args.target_group} beta={args.beta}.")
    rows = []
    for row in status.itertuples(index=False):
        run_dir = Path(str(row.run_dir))
        checkpoint = run_dir / args.checkpoint_name
        metrics = run_dir / "metrics.csv"
        if not checkpoint.exists() or not metrics.exists():
            continue
        metric_row = summarize_train_metrics(metrics)
        rows.append({**row._asdict(), "checkpoint": str(checkpoint), **metric_row})
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise ValueError("No completed runs with checkpoint and metrics.csv.")
    alive = candidates[candidates["recent_kl_per_instance"].ge(args.min_kl_per_instance)].copy()
    if alive.empty:
        alive = candidates.copy()
        alive["selection_note"] = "fallback_no_kl_alive"
    else:
        alive["selection_note"] = "kl_alive"
    alive = alive.sort_values(["recent_rc_per_instance", "recent_kl_per_instance"], ascending=[True, False])
    selected = alive.iloc[0].to_dict()
    return Path(str(selected["checkpoint"])), selected


def summarize_train_metrics(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["loss_kl_per_instance"] = pd.to_numeric(df["loss_kl_per_instance"], errors="coerce")
    df["loss_reconstruction_per_instance"] = pd.to_numeric(
        df["loss_reconstruction_per_instance"],
        errors="coerce",
    )
    train = df[df["split"].eq("train")].dropna(subset=["step"]).sort_values("step")
    max_step = int(train["step"].max())
    recent = train[train["step"].ge(max_step - 200)]
    return {
        "max_step": max_step,
        "recent_kl_per_instance": float(recent["loss_kl_per_instance"].mean()),
        "recent_rc_per_instance": float(recent["loss_reconstruction_per_instance"].mean()),
    }


@torch.no_grad()
def extract_prior_mu(args: argparse.Namespace, checkpoint: Path) -> tuple[np.ndarray, np.ndarray]:
    config_args = argparse.Namespace(checkpoint=checkpoint)
    config, _ = load_checkpoint_config_and_schema(config_args)
    config["data"]["target_group"] = args.target_group
    config["data"]["dataset_name"] = args.dataset_name
    config["data"]["dataset_root"] = args.dataset_root
    config["data"].pop("max_rows_per_split", None)
    schema = load_grouped_cvae_schema(config)
    model = build_model(config, schema)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=True)
    device = resolve_device(args.device)
    model.to(device).eval()
    loader = create_grouped_cvae_dataloader(
        config,
        split=args.split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        seed=args.seed,
    )
    parts = []
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        condition = model.encode_condition(
            source_num=batch["source_num"],
            source_cat=batch["source_cat"],
            source_num_mask=batch["source_num_mask"],
            source_cat_mask=batch["source_cat_mask"],
        )
        prior_mu, _ = model.encode_prior(condition)
        parts.append(prior_mu.detach().float().cpu().numpy())
    latents = np.concatenate(parts, axis=0)
    row_indices = split_row_indices(args)
    if len(row_indices) != len(latents):
        raise ValueError(f"row_indices length {len(row_indices)} != latent length {len(latents)}")
    return latents, row_indices


def split_row_indices(args: argparse.Namespace) -> np.ndarray:
    split_path = Path(args.dataset_root) / args.dataset_name / args.target_group / "split.csv"
    indices = []
    with split_path.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["split"] == args.split:
                indices.append(int(row["row_index"]))
    return np.asarray(indices, dtype=np.int64)


def source_labels(args: argparse.Namespace, row_indices: np.ndarray) -> np.ndarray:
    meta = pd.read_csv(
        args.harmonized_csv,
        usecols=["source", "source_year"],
        encoding="utf-8-sig",
    )
    selected = meta.iloc[row_indices].copy()
    source = selected["source"].astype(str).str.upper()
    year = pd.to_numeric(selected["source_year"], errors="coerce")
    labels = np.full(len(selected), "other", dtype=object)
    labels[(source.eq("KNHANES")) & year.ge(2013) & year.le(2024)] = "KNHANES 2013-2024"
    labels[(source.eq("KNHANES")) & year.ge(1998) & year.lt(2013)] = "KNHANES 1998-2012"
    labels[(source.eq("NHANES")) & year.ge(1988) & year.le(2023)] = "NHANES 1988-2023"
    return labels


def finite_label_mask(labels: np.ndarray) -> np.ndarray:
    return np.isin(labels, CLASS_ORDER)


def stratified_sample(labels: np.ndarray, *, max_points: int, seed: int, mode: str) -> np.ndarray:
    n = len(labels)
    if max_points <= 0 or n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    chosen = []
    if mode == "balanced":
        per_class = max(1, max_points // len(CLASS_ORDER))
        for label in CLASS_ORDER:
            idx = np.flatnonzero(labels == label)
            quota = min(per_class, len(idx))
            chosen.append(rng.choice(idx, size=quota, replace=False))
        return np.sort(np.concatenate(chosen))
    for label in CLASS_ORDER:
        idx = np.flatnonzero(labels == label)
        quota = max(1, int(round(max_points * len(idx) / n)))
        quota = min(quota, len(idx))
        chosen.append(rng.choice(idx, size=quota, replace=False))
    out = np.concatenate(chosen)
    if len(out) > max_points:
        out = rng.choice(out, size=max_points, replace=False)
    return np.sort(out)


def run_umap(latents: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    try:
        import umap
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "UMAP requires the optional package 'umap-learn'. Install with: uv add umap-learn"
        ) from exc
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed,
    )
    return reducer.fit_transform(latents).astype(np.float64)


def plot_matplotlib(embedding: np.ndarray, labels: np.ndarray, path: Path, args: argparse.Namespace) -> None:
    fig = plt.figure(figsize=(10.5, 8.2), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    plot_order = labels_by_descending_count(labels)
    for label in plot_order:
        mask = labels == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],
            s=7,
            alpha=0.38,
            color=CLASS_COLORS[label],
            label=f"{label} (n={int(mask.sum())})",
            depthshade=False,
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    ax.set_title(f"{args.target_group} train conditional-prior latent UMAP", fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_pairwise_projections(embedding: np.ndarray, labels: np.ndarray, path: Path, args: argparse.Namespace) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=180)
    pairs = [(0, 1), (0, 2), (1, 2)]
    plot_order = labels_by_descending_count(labels)
    for ax, (x_idx, y_idx) in zip(axes, pairs, strict=True):
        for label in plot_order:
            mask = labels == label
            ax.scatter(
                embedding[mask, x_idx],
                embedding[mask, y_idx],
                s=5,
                alpha=0.24,
                color=CLASS_COLORS[label],
                label=f"{label} (n={int(mask.sum())})",
                linewidths=0,
            )
        ax.set_xlabel(f"UMAP {x_idx + 1}")
        ax.set_ylabel(f"UMAP {y_idx + 1}")
        ax.grid(True, alpha=0.18, linewidth=0.7)
    handles, labels_text = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_text, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        f"{args.target_group} train conditional-prior latent UMAP projections",
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_plotly(embedding: np.ndarray, labels: np.ndarray, path: Path, args: argparse.Namespace) -> None:
    import plotly.graph_objects as go

    fig = go.Figure()
    plot_order = labels_by_descending_count(labels)
    for label in plot_order:
        mask = labels == label
        fig.add_trace(
            go.Scatter3d(
                x=embedding[mask, 0],
                y=embedding[mask, 1],
                z=embedding[mask, 2],
                mode="markers",
                name=f"{label} (n={int(mask.sum())})",
                marker={"size": 2.4, "opacity": 0.34, "color": CLASS_COLORS[label]},
            )
        )
    fig.update_layout(
        title=f"{args.target_group} train conditional-prior latent UMAP",
        scene={"xaxis_title": "UMAP 1", "yaxis_title": "UMAP 2", "zaxis_title": "UMAP 3"},
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
    )
    fig.write_html(path)


def labels_by_descending_count(labels: np.ndarray) -> list[str]:
    return sorted(CLASS_ORDER, key=lambda label: int(np.sum(labels == label)), reverse=True)


def class_counts(labels: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        [{"source_class": label, "n": int(np.sum(labels == label))} for label in CLASS_ORDER]
    )


def default_output_dir(args: argparse.Namespace) -> Path:
    root_name = args.sweep_root.name if args.sweep_root is not None else "manual_checkpoint"
    return Path("outputs/analysis") / root_name / "train_prior_umap3d_by_source"


def default_figures_dir(args: argparse.Namespace) -> Path:
    root_name = args.sweep_root.name if args.sweep_root is not None else "manual_checkpoint"
    return Path("figures") / root_name / "train_prior_umap3d_by_source"


if __name__ == "__main__":
    main()
