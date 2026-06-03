"""Render the CVAE beta-selection experiment hyperparameters as a slide table."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from matplotlib.patches import Rectangle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train/train_conditional_vae_eddi_z16_beta010.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="cvae_config_hyperparams")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def draw_section(ax, x: float, y: float, w: float, title: str, rows: list[tuple[str, str]]) -> float:
    row_h = 0.034
    header_h = 0.045
    h = header_h + row_h * len(rows)

    ax.add_patch(Rectangle((x, y - h), w, h, facecolor="#FFFFFF", edgecolor="#D1D5DB", linewidth=0.9))
    ax.add_patch(Rectangle((x, y - header_h), w, header_h, facecolor="#F3F4F6", edgecolor="none"))
    ax.text(x + 0.015, y - header_h / 2, title, fontsize=11.5, fontweight="bold", color="#111827", va="center")

    key_w = w * 0.42
    for idx, (key, value) in enumerate(rows):
        top = y - header_h - idx * row_h
        bg = "#FFFFFF" if idx % 2 == 0 else "#FAFAFA"
        ax.add_patch(Rectangle((x, top - row_h), w, row_h, facecolor=bg, edgecolor="#E5E7EB", linewidth=0.45))
        ax.text(x + 0.014, top - row_h / 2, key, fontsize=8.9, color="#374151", va="center")
        ax.text(x + key_w + 0.014, top - row_h / 2, value, fontsize=8.9, color="#111827", va="center")

    return h


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text())
    model_cfg = config["model"]
    train_cfg = config["train"]
    source_cfg = model_cfg["source_encoder"]

    def list_value(values: object) -> str:
        if isinstance(values, list):
            return "[" + ", ".join(str(value) for value in values) + "]"
        return str(values)

    sections = [
        (
            "Model Architecture",
            [
                ("Backbone", "EDDI CVAE"),
                (
                    "Source encoder",
                    f"embedding {source_cfg['embedding_dim']} -> MLP {list_value(source_cfg['hidden_layers'])} -> output {source_cfg['output_dim']}",
                ),
                ("Source aggregation", str(source_cfg.get("aggregation", "sum"))),
                ("Latent dim", str(model_cfg["latent_dim"])),
                ("Encoder MLP", list_value(model_cfg["encoder_hidden_layers"])),
                ("Prior MLP", list_value(model_cfg["prior_hidden_layers"])),
                ("Decoder MLP", list_value(model_cfg["decoder_hidden_layers"])),
                ("Activation / dropout", f"{model_cfg.get('activation', 'ReLU')} / {model_cfg['dropout']}"),
                ("Normalization", "BatchNorm enabled" if model_cfg.get("batch_norm") else "BatchNorm disabled"),
            ],
        ),
        (
            "Training Hyperparameters",
            [
                ("Steps", f"{train_cfg['steps']} fixed steps"),
                ("Batch size", str(train_cfg["batch_size"])),
                ("Learning rate", f"{train_cfg['learning_rate']:g}"),
                ("Weight decay", f"{train_cfg['weight_decay']:g}"),
                ("Beta warmup", f"{train_cfg['beta_warmup_steps']} steps"),
                ("Early stopping", "disabled" if not train_cfg.get("early_stopping", {}).get("enabled", False) else "enabled"),
                ("Gradient clipping", str(train_cfg.get("gradient_clip_norm"))),
                ("Masked target dropout", str(train_cfg.get("masked_target_dropout", 0.0))),
                ("Mixed precision", str(train_cfg.get("mixed_precision", False)).lower()),
                ("Num workers", str(train_cfg.get("num_workers", 0))),
            ],
        ),
    ]

    fig = plt.figure(figsize=(13.33, 7.5), dpi=args.dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    ax.text(0.04, 0.945, "CVAE hyperparameters", fontsize=25, fontweight="bold", color="#18202A", va="top")
    ax.text(
        0.04,
        0.900,
        "Model and training settings used in the beta-selection sweep.",
        fontsize=10.5,
        color="#4B5563",
        va="top",
    )

    left_x = 0.08
    right_x = 0.525
    panel_w = 0.395
    top_y = 0.835

    draw_section(ax, left_x, top_y, panel_w, sections[0][0], sections[0][1])
    draw_section(ax, right_x, top_y, panel_w, sections[1][0], sections[1][1])

    ax.text(
        0.04,
        0.045,
        f"Config base: {args.config}. "
        "Runner overrides beta, batch size, beta warmup, and early stopping.",
        fontsize=8.8,
        color="#6B7280",
        va="bottom",
    )

    png = args.output_dir / f"{args.output_prefix}.png"
    svg = args.output_dir / f"{args.output_prefix}.svg"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    csv = args.output_dir / f"{args.output_prefix}.csv"
    with csv.open("w", encoding="utf-8") as f:
        f.write("section,parameter,value\n")
        for section, rows in sections:
            for key, value in rows:
                f.write(f'"{section}","{key}","{value}"\n')

    print(f"wrote {csv}")
    print(f"wrote {png}")
    print(f"wrote {svg}")


if __name__ == "__main__":
    main()
