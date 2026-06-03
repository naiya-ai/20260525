#!/usr/bin/env python
"""Summarize masked-AE downstream classifier runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    args = parse_args()
    rows = []
    for summary_path in sorted(args.input_root.glob("*/summary.json")):
        run_dir = summary_path.parent
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config_path = run_dir / "config.yaml"
        row = {
            "run": run_dir.name,
            "run_dir": str(run_dir),
            **summary,
        }
        if config_path.exists():
            import yaml

            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            row["variant"] = config.get("model", {}).get("variant_name", run_dir.name)
            row["target_disease"] = config.get("data", {}).get("target_disease")
            row["class_weight"] = config.get("model", {}).get("class_weight")
            row["freeze_backbone"] = config.get("model", {}).get("freeze_backbone")
            row["pretrained_autoencoder_path"] = config.get("model", {}).get("pretrained_autoencoder_path")
        rows.append(row)
    frame = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    if not frame.empty:
        sort_col = "best_valid_auroc" if "best_valid_auroc" in frame.columns else "test_auroc"
        print(frame.sort_values(sort_col, ascending=False).to_string(index=False))
    print(f"wrote {args.output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--output-csv", type=Path, default=Path("figures/masked_ae_downstream_summary.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    main()
