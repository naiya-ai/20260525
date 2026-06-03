"""Evaluate all diseases from one CVAE trained on all disease indicators."""

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
    build_model,
    decode_source_context,
    load_checkpoint_config_and_schema,
    parse_label,
    read_target_metadata,
    resolve_device,
    set_seed,
)
from eval.evaluate_cvae_prior_probability import (  # noqa: E402
    decoded_sample_disease_probabilities,
    probabilities_to_predictions,
    select_threshold,
)
from train.dataset import create_grouped_cvae_dataloader, load_grouped_cvae_schema  # noqa: E402


def main() -> None:
    args = parse_args()
    checkpoint_config, _ = load_checkpoint_config_and_schema(args)
    config = checkpoint_config
    config["data"]["target_group"] = args.target_group
    if args.dataset_name is not None:
        config["data"]["dataset_name"] = args.dataset_name
    if args.dataset_root is not None:
        config["data"]["dataset_root"] = args.dataset_root
    if args.max_rows is not None:
        config["data"]["max_rows_per_split"] = args.max_rows

    device = resolve_device(args.device)
    set_seed(args.seed)
    schema = load_grouped_cvae_schema(config)
    target_metadata = read_target_metadata(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(config, schema)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    valid = evaluate_split(
        model=model,
        config=config,
        schema=schema,
        target_metadata=target_metadata,
        split="valid",
        diseases=args.diseases,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=device,
    )
    test = evaluate_split(
        model=model,
        config=config,
        schema=schema,
        target_metadata=target_metadata,
        split="test",
        diseases=args.diseases,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=device,
    )

    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for disease in args.diseases:
        threshold_result = select_threshold(
            probabilities=valid[disease]["probabilities"],
            labels=valid[disease]["labels"],
            strategy=args.threshold_strategy,
            fixed_threshold=args.threshold,
        )
        predictions = probabilities_to_predictions(
            test[disease]["probabilities"],
            threshold=threshold_result["threshold"],
        )
        metrics = {
            "variant": args.variant,
            "model": "cvae",
            "target_group": args.target_group,
            "disease": disease,
            "dataset_name": config["data"]["dataset_name"],
            "checkpoint": str(args.checkpoint),
            "checkpoint_step": int(checkpoint.get("step", -1)),
            "num_samples": args.num_samples,
            "threshold": threshold_result["threshold"],
            "auroc": binary_auroc(test[disease]["probabilities"], test[disease]["labels"]),
            **binary_metrics(predictions, test[disease]["labels"]),
        }
        rows.append(metrics)
        pd.DataFrame(
            {
                "row_id": test[disease]["row_ids"],
                "y_true": test[disease]["labels"],
                "disease_probability": test[disease]["probabilities"],
                "prediction": predictions,
            }
        ).to_csv(args.output_dir / f"{args.variant}_{disease}_probabilities.csv", index=False)

    summary = pd.DataFrame(rows)
    summary_path = args.output_dir / f"{args.variant}_joint_disease_metrics.csv"
    json_path = args.output_dir / f"{args.variant}_joint_disease_metrics.json"
    summary.to_csv(summary_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(f"saved {summary_path}", flush=True)
    print(f"saved {json_path}", flush=True)


@torch.no_grad()
def evaluate_split(
    *,
    model: Any,
    config: dict[str, Any],
    schema: dict[str, Any],
    target_metadata: dict[str, Any],
    split: str,
    diseases: list[str],
    num_samples: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    loader = create_grouped_cvae_dataloader(
        config,
        split=split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
    )
    row_ids = split_row_ids(config, split=split)
    labels_by_disease = {disease: read_labels(config, disease, row_ids) for disease in diseases}
    out = {
        disease: {
            "row_ids": row_ids,
            "labels": labels_by_disease[disease],
            "probabilities": [],
        }
        for disease in diseases
    }

    row_offset = 0
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        batch_size_actual = int(batch["source_num"].shape[0])
        condition = model.encode_condition(
            source_num=batch["source_num"],
            source_cat=batch["source_cat"],
            source_num_mask=batch["source_num_mask"],
            source_cat_mask=batch["source_cat_mask"],
        )
        source_context = decode_source_context(batch=batch, schema=schema, config=config)
        positive = {disease: np.zeros(batch_size_actual, dtype=np.float64) for disease in diseases}
        observed = {disease: np.zeros(batch_size_actual, dtype=np.float64) for disease in diseases}
        for _ in range(num_samples):
            z = model.sample_prior(condition, sample=True)
            if hasattr(model, "decode_from_source"):
                decoded = model.decode_from_source(
                    z,
                    source_num=batch["source_num"],
                    source_cat=batch["source_cat"],
                    source_num_mask=batch["source_num_mask"],
                    source_cat_mask=batch["source_cat_mask"],
                    condition=condition,
                )
            else:
                decoded = model.decode(z, condition)
            for disease in diseases:
                values = decoded_sample_disease_probabilities(
                    decoded=decoded,
                    target_metadata=target_metadata,
                    disease=disease,
                    source_context=source_context,
                )
                for idx, value in enumerate(values):
                    if value is None:
                        continue
                    observed[disease][idx] += 1.0
                    positive[disease][idx] += float(value)
        for disease in diseases:
            for pos, count in zip(positive[disease], observed[disease]):
                out[disease]["probabilities"].append(None if count == 0 else float(pos / count))
        row_offset += batch_size_actual

    if row_offset != len(row_ids):
        raise ValueError(f"Processed {row_offset} rows but split has {len(row_ids)} row ids.")
    return out


def split_row_ids(config: dict[str, Any], *, split: str) -> np.ndarray:
    path = (
        Path(config["data"]["dataset_root"])
        / str(config["data"]["dataset_name"])
        / str(config["data"]["target_group"])
        / "split.csv"
    )
    rows = []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["split"] == split:
                rows.append(int(row["row_index"]))
    if config["data"].get("max_rows_per_split") is not None:
        rows = rows[: int(config["data"]["max_rows_per_split"])]
    return np.asarray(rows, dtype=np.int64)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-group", default="all_disease_indicators")
    parser.add_argument("--variant", default="cvae_all_indicators")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--dataset-name")
    parser.add_argument("--diseases", nargs="+", default=list(DISEASES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument(
        "--threshold-strategy",
        choices=["fixed", "validation_balanced_accuracy"],
        default="validation_balanced_accuracy",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    main()
