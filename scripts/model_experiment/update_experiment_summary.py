"""Update CVAE model-experiment summaries and figures after one run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import build_conditional_vae_model  # noqa: E402
from train import load_grouped_cvae_schema  # noqa: E402


SUMMARY_COLUMNS = [
    "model",
    "width",
    "rep",
    "seed",
    "run_dir",
    "config",
    "checkpoint",
    "metrics_json",
    "parameters",
    "forward_flops_per_sample",
    "best_valid_loss",
    "final_valid_loss",
    "test_auroc",
    "n_evaluable",
]


def main() -> None:
    args = parse_args()
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.loss_dir.mkdir(parents=True, exist_ok=True)

    config = load_yaml(args.config)
    schema = load_grouped_cvae_schema(config)
    parameters = count_parameters(config, schema)
    flops = estimate_forward_flops_per_sample(config, schema)
    loss_summary = summarize_metrics(args.run_dir / "metrics.csv")
    eval_metrics = load_json(args.metrics_json)

    row = {
        "model": args.model,
        "width": str(args.width),
        "rep": str(args.rep),
        "seed": str(config.get("seed", "")),
        "run_dir": str(args.run_dir),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "metrics_json": str(args.metrics_json),
        "parameters": str(parameters),
        "forward_flops_per_sample": str(flops),
        "best_valid_loss": format_float(loss_summary["best_valid_loss"]),
        "final_valid_loss": format_float(loss_summary["final_valid_loss"]),
        "test_auroc": format_float(eval_metrics.get("auroc")),
        "n_evaluable": str(eval_metrics.get("n_evaluable", "")),
    }
    upsert_summary(args.summary_csv, row)
    plot_loss_curve(
        metrics_csv=args.run_dir / "metrics.csv",
        output_path=args.loss_dir / f"{args.model}_width{args.width}_rep{args.rep:02d}_loss.png",
        title=f"{args.model} width={args.width} rep={args.rep}",
    )
    rows = read_summary(args.summary_csv)
    plot_scatter(
        rows=rows,
        x_key="parameters",
        y_key="test_auroc",
        output_path=args.figures_dir / "auroc_vs_parameters.png",
        xlabel="Parameters",
        ylabel="Test AUROC",
        title="AUROC vs Parameters",
    )
    plot_scatter(
        rows=rows,
        x_key="forward_flops_per_sample",
        y_key="test_auroc",
        output_path=args.figures_dir / "auroc_vs_compute.png",
        xlabel="Estimated forward FLOPs / sample",
        ylabel="Test AUROC",
        title="AUROC vs Compute",
    )
    plot_scatter(
        rows=rows,
        x_key="parameters",
        y_key="best_valid_loss",
        output_path=args.figures_dir / "loss_vs_parameters.png",
        xlabel="Parameters",
        ylabel="Best validation loss",
        title="Loss vs Parameters",
    )
    plot_scatter(
        rows=rows,
        x_key="forward_flops_per_sample",
        y_key="best_valid_loss",
        output_path=args.figures_dir / "loss_vs_compute.png",
        xlabel="Estimated forward FLOPs / sample",
        ylabel="Best validation loss",
        title="Loss vs Compute",
    )
    print(json.dumps(row, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument("--loss-dir", type=Path, required=True)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return data


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def count_parameters(config: dict[str, Any], schema: dict[str, Any]) -> int:
    model = build_conditional_vae_model(
        source_n_num_features=int(schema["source"]["n_num_features"]),
        source_category_sizes=schema["source"]["category_sizes"],
        target_n_num_features=int(schema["target"]["n_num_features"]),
        target_category_sizes=schema["target"]["category_sizes"],
        model_config=config["model"],
    )
    return int(sum(parameter.numel() for parameter in model.parameters()))


def estimate_forward_flops_per_sample(config: dict[str, Any], schema: dict[str, Any]) -> int:
    model_config = config["model"]
    backbone = str(model_config.get("backbone", "eddi")).lower()
    source_num = int(schema["source"]["n_num_features"])
    source_cats = [int(size) for size in schema["source"]["category_sizes"]]
    target_num = int(schema["target"]["n_num_features"])
    target_cats = [int(size) for size in schema["target"]["category_sizes"]]
    latent_dim = int(model_config["latent_dim"])
    target_cat_dim = sum(target_cats)
    target_context_dim = target_num + target_cat_dim + target_num + len(target_cats)
    target_output_dim = target_num + target_cat_dim
    if backbone == "raw_mlp":
        condition_dim = 2 * source_num + sum(source_cats) + len(source_cats)
        encoder_hidden = [int(x) for x in model_config.get("encoder_hidden_layers", [])]
        prior_hidden = [int(x) for x in model_config.get("prior_hidden_layers", encoder_hidden)]
        decoder_hidden = [int(x) for x in model_config.get("decoder_hidden_layers", [])]
        return int(
            mlp_flops([condition_dim + target_context_dim, *encoder_hidden, 2 * latent_dim])
            + mlp_flops([condition_dim, *prior_hidden, 2 * latent_dim])
            + mlp_flops([condition_dim + latent_dim, *decoder_hidden, target_output_dim])
        )
    if backbone == "transformer":
        d_model = int(model_config.get("d_model", 128))
        source_tokens = source_num + len(source_cats)
        target_tokens = target_num + len(target_cats)
        encoder_cfg = model_config.get("encoder", {})
        prior_cfg = model_config.get("prior", encoder_cfg)
        decoder_cfg = model_config.get("decoder", encoder_cfg)
        return int(
            transformer_stack_flops(source_tokens + target_tokens + 1, d_model, encoder_cfg)
            + transformer_stack_flops(source_tokens + 1, d_model, prior_cfg)
            + transformer_stack_flops(source_tokens + 1 + target_tokens, d_model, decoder_cfg)
            + 2 * d_model * 2 * latent_dim
            + 2 * latent_dim * d_model
            + 2 * d_model * target_output_dim
        )
    return count_linear_module_flops(config, schema)


def mlp_flops(dims: list[int]) -> int:
    return int(sum(2 * left * right for left, right in zip(dims, dims[1:])))


def transformer_stack_flops(seq_len: int, d_model: int, cfg: Any) -> int:
    cfg = cfg if isinstance(cfg, dict) else {}
    n_layers = int(cfg.get("n_layers", 1))
    ff_dim = int(cfg.get("ff_dim", 4 * d_model))
    per_layer = (
        8 * seq_len * d_model * d_model
        + 4 * seq_len * seq_len * d_model
        + 4 * seq_len * d_model * ff_dim
    )
    return int(n_layers * per_layer)


def count_linear_module_flops(config: dict[str, Any], schema: dict[str, Any]) -> int:
    model = build_conditional_vae_model(
        source_n_num_features=int(schema["source"]["n_num_features"]),
        source_category_sizes=schema["source"]["category_sizes"],
        target_n_num_features=int(schema["target"]["n_num_features"]),
        target_category_sizes=schema["target"]["category_sizes"],
        model_config=config["model"],
    )
    total = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            total += 2 * int(module.in_features) * int(module.out_features)
    return int(total)


def summarize_metrics(metrics_csv: Path) -> dict[str, float | None]:
    if not metrics_csv.exists():
        return {"best_valid_loss": None, "final_valid_loss": None}
    valid_losses: list[float] = []
    with metrics_csv.open(newline="") as file:
        for row in csv.DictReader(file):
            if row.get("split") != "valid":
                continue
            value = parse_float(row.get("loss"))
            if value is not None:
                valid_losses.append(value)
    return {
        "best_valid_loss": min(valid_losses) if valid_losses else None,
        "final_valid_loss": valid_losses[-1] if valid_losses else None,
    }


def upsert_summary(path: Path, row: dict[str, str]) -> None:
    rows = read_summary(path) if path.exists() else []
    key = (row["model"], row["width"], row["rep"])
    rows = [
        existing
        for existing in rows
        if (existing.get("model"), existing.get("width"), existing.get("rep")) != key
    ]
    rows.append(row)
    rows.sort(key=lambda item: (item.get("model", ""), int(item.get("width", 0)), int(item.get("rep", 0))))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def plot_loss_curve(metrics_csv: Path, output_path: Path, title: str) -> None:
    if not metrics_csv.exists():
        return
    series: dict[str, tuple[list[int], list[float]]] = {
        "train": ([], []),
        "valid": ([], []),
    }
    with metrics_csv.open(newline="") as file:
        for row in csv.DictReader(file):
            split = row.get("split")
            value = parse_float(row.get("loss"))
            step = row.get("step")
            if split not in series or value is None or step is None:
                continue
            series[split][0].append(int(step))
            series[split][1].append(value)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for split, (steps, values) in series.items():
        if steps:
            ax.plot(steps, values, marker="o", markersize=2, linewidth=1.2, label=split)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_ylim(0.0, 0.1)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_scatter(
    *,
    rows: list[dict[str, str]],
    x_key: str,
    y_key: str,
    output_path: Path,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    points = []
    for row in rows:
        x = parse_float(row.get(x_key))
        y = parse_float(row.get(y_key))
        width = parse_float(row.get("width"))
        if x is None or y is None or width is None:
            continue
        points.append((x, y, int(width), row.get("model", "")))
    fig, ax = plt.subplots(figsize=(7, 4.8))
    if points:
        widths = sorted({point[2] for point in points})
        cmap = plt.get_cmap("viridis")
        color_by_width = {
            width: cmap(index / max(1, len(widths) - 1))
            for index, width in enumerate(widths)
        }
        for width in widths:
            subset = [point for point in points if point[2] == width]
            ax.scatter(
                [point[0] for point in subset],
                [point[1] for point in subset],
                s=42,
                alpha=0.82,
                color=color_by_width[width],
                label=f"width {width}",
            )
        ax.set_xscale("log")
        ax.legend(fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def format_float(value: Any) -> str:
    parsed = parse_float(value)
    return "" if parsed is None else f"{parsed:.12g}"


if __name__ == "__main__":
    main()
