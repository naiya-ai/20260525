"""Generate DDPM all-indicator samples and evaluate every disease."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.cvae_common import (  # noqa: E402
    DISEASES,
    binary_auroc,
    binary_metrics,
    decode_source_context,
    disease_predictions,
    parse_label,
)
from eval.ddpm_common import (  # noqa: E402
    load_ddpm_checkpoint_config_schema,
    load_target_metadata,
    restore_target_values,
    split_row_ids_for_loaded_dataset,
)
from eval.evaluate_cvae_coverage import write_feature_interval_csv  # noqa: E402
from scripts.eval.evaluate_ddpm_prior_probability import (  # noqa: E402
    evaluate_coverage_from_saved_samples,
    sample_targets,
)
from scripts.train.train_ddpm_mlp import build_model_stack  # noqa: E402
from train.dataset import load_grouped_cvae_dataset  # noqa: E402


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    checkpoint, config, schema = load_ddpm_checkpoint_config_schema(args.checkpoint)
    config["data"]["target_group"] = args.target_group
    if args.dataset_name is not None:
        config["data"]["dataset_name"] = args.dataset_name
    if args.dataset_root is not None:
        config["data"]["dataset_root"] = args.dataset_root
    if args.max_rows is not None:
        config["data"]["max_rows_per_split"] = args.max_rows

    target_metadata = load_target_metadata(config)
    dataset = load_grouped_cvae_dataset(config, split=args.split)
    row_ids = split_row_ids_for_loaded_dataset(config, split=args.split)
    if len(row_ids) != len(dataset):
        raise ValueError(f"row_ids ({len(row_ids)}) != loaded dataset ({len(dataset)}).")

    source_encoder, diffusion = build_model_stack(config, schema)
    diffusion.denoise_fn.load_state_dict(checkpoint["model_state_dict"])
    source_state = checkpoint.get("source_encoder_state_dict")
    if source_encoder is not None and source_state is not None:
        source_encoder.load_state_dict(source_state)

    device = resolve_device(args.device)
    diffusion.to(device).eval()
    if source_encoder is not None:
        source_encoder.to(device).eval()

    sample_paths = []
    for sample_idx in range(1, args.num_samples + 1):
        sample_path = samples_dir / f"sample_{sample_idx:03d}.csv"
        sample_paths.append(sample_path)
        if sample_path.exists() and not args.overwrite_samples:
            continue
        torch.manual_seed(args.seed + sample_idx - 1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + sample_idx - 1)
        generated = sample_targets(
            diffusion=diffusion,
            source_encoder=source_encoder,
            dataset=dataset,
            batch_size=args.batch_size,
            device=device,
        )
        restored = restore_target_values(
            target_metadata=target_metadata,
            num=generated["num"],
            cat=generated["cat"],
        )
        frame = pd.DataFrame(restored)
        frame.insert(0, "sample", sample_idx)
        frame.insert(0, "row_id", row_ids)
        frame.to_csv(sample_path, index=False)
        print(f"saved sample {sample_idx}: {sample_path}", flush=True)

    source_context = decode_source_context(
        batch={
            "source_num": dataset.source_num,
            "source_cat": dataset.source_cat,
            "source_num_mask": dataset.source_num_mask,
            "source_cat_mask": dataset.source_cat_mask,
        },
        schema=schema,
        config=config,
    )
    rows = []
    for disease in args.diseases:
        scores = score_one_disease(
            disease=disease,
            sample_paths=sample_paths,
            source_context=source_context,
            labels=read_labels(config, disease, row_ids),
        )
        probabilities = [None if pd.isna(v) else float(v) for v in scores["disease_probability"]]
        labels = [None if pd.isna(v) else bool(v) for v in scores["y_true"]]
        predictions = [None if v is None else bool(v >= args.threshold) for v in probabilities]
        metrics = {
            "variant": args.variant,
            "model": "ddpm",
            "target_group": args.target_group,
            "disease": disease,
            "dataset_name": config["data"]["dataset_name"],
            "checkpoint": str(args.checkpoint),
            "checkpoint_step": int(checkpoint.get("step", -1)),
            "best_valid_loss": float(checkpoint.get("best_valid_loss", float("nan"))),
            "num_timesteps": int(config["diffusion"]["num_timesteps"]),
            "num_samples": args.num_samples,
            "threshold": args.threshold,
            "auroc": binary_auroc(probabilities, labels),
            **binary_metrics(predictions, labels),
        }
        rows.append(metrics)
        scores.to_csv(args.output_dir / f"{args.variant}_{disease}_scores.csv", index=False)

    coverage = evaluate_coverage_from_saved_samples(
        sample_paths=sample_paths,
        dataset=dataset,
        target_metadata=target_metadata,
        interval_levels=args.interval_levels,
    )
    coverage_output = {
        "checkpoint": str(args.checkpoint),
        "variant": args.variant,
        "dataset_name": config["data"]["dataset_name"],
        "target_group": args.target_group,
        "source_groups": list(config["data"]["source_groups"]),
        "split": args.split,
        "num_prior_samples": args.num_samples,
        "interval_levels": args.interval_levels,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "mixed_precision": False,
        "prior_sample_cache": str(samples_dir),
        "used_sample_cache": True,
        **coverage,
    }
    summary = pd.DataFrame(rows)
    summary_path = args.output_dir / f"{args.variant}_joint_disease_metrics.csv"
    summary.to_csv(summary_path, index=False)
    (args.output_dir / f"{args.variant}_joint_disease_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output_dir / f"{args.variant}_coverage.json").write_text(json.dumps(coverage_output, indent=2) + "\n")
    write_feature_interval_csv(args.output_dir / f"{args.variant}_coverage.csv", coverage_output)
    print(summary.to_string(index=False), flush=True)
    print(f"saved {summary_path}", flush=True)


def score_one_disease(
    *,
    disease: str,
    sample_paths: list[Path],
    source_context: dict[str, np.ndarray],
    labels: list[bool | None],
) -> pd.DataFrame:
    row_ids = None
    positive = None
    observed = None
    for sample_path in sample_paths:
        frame = pd.read_csv(sample_path)
        current_row_ids = frame["row_id"].to_numpy(dtype=np.int64)
        if row_ids is None:
            row_ids = current_row_ids
            positive = np.zeros(len(row_ids), dtype=np.float64)
            observed = np.zeros(len(row_ids), dtype=np.float64)
        elif not np.array_equal(row_ids, current_row_ids):
            raise ValueError(f"row_id order differs in {sample_path}.")
        target = {
            column: frame[column].to_numpy()
            for column in frame.columns
            if column not in {"row_id", "sample"}
        }
        predictions = disease_predictions(disease, target, source_context)
        for idx, value in enumerate(predictions):
            if value is None:
                continue
            observed[idx] += 1.0
            positive[idx] += float(value)
    if row_ids is None or positive is None or observed is None:
        raise ValueError("No samples were provided.")
    probabilities = [None if count == 0 else pos / count for pos, count in zip(positive, observed)]
    return pd.DataFrame(
        {
            "row_id": row_ids,
            "y_true": labels,
            "disease_probability": probabilities,
            "positive_samples": positive,
            "observed_samples": observed,
            "num_samples": len(sample_paths),
        }
    )


def read_labels(config: dict[str, Any], disease: str, row_ids: np.ndarray) -> list[bool | None]:
    dataset_name = str(config["data"]["dataset_name"])
    suffix = dataset_name.removeprefix("harmonized_")
    path = Path("datasets/harmonized") / f"disease_labels_harmonized_{suffix}.csv"
    column = f"disease_{disease}"
    labels = {}
    with path.open(newline="", errors="replace") as file:
        reader = csv.DictReader(file)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"{path} does not contain {column}.")
        for row in reader:
            labels[int(row["row_index"])] = parse_label(row[column])
    return [labels[int(idx)] for idx in row_ids]


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-group", default="all_disease_indicators")
    parser.add_argument("--variant", default="ddpm_all_indicators")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--dataset-name")
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--diseases", nargs="+", default=list(DISEASES))
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--interval-levels", nargs="+", type=float, default=[0.9, 0.95, 0.99])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--overwrite-samples", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
