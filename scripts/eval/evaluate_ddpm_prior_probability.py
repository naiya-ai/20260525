"""Generate DDPM test samples and evaluate disease probabilities."""

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

from eval.cvae_common import binary_auroc, binary_metrics, decode_source_context, disease_predictions, parse_label
from eval.ddpm_common import (
    load_ddpm_checkpoint_config_schema,
    load_target_metadata,
    restore_target_values,
    split_row_ids_for_loaded_dataset,
)
from eval.evaluate_cvae_coverage import (
    FeatureCoverageAccumulator,
    aggregate_feature_results,
    write_feature_interval_csv,
)
from scripts.train.train_ddpm_mlp import build_model_stack
from train.dataset import load_grouped_cvae_dataset


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    checkpoint, config, schema = load_ddpm_checkpoint_config_schema(args.checkpoint)
    disease = str(config["data"]["target_group"])
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

    sample_paths = []
    for sample_idx in range(1, args.num_samples + 1):
        sample_path = samples_dir / f"sample_{sample_idx:03d}.csv"
        sample_paths.append(sample_path)
        if sample_path.exists() and not args.overwrite_samples:
            print(f"skip existing sample {sample_idx}: {sample_path}", flush=True)
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

    scores = score_samples(
        disease=disease,
        sample_paths=sample_paths,
        source_context=source_context,
        labels=read_labels(config, disease, row_ids),
    )
    scores_path = output_dir / f"{disease}_ddpm_scores.csv"
    metrics_path = output_dir / f"{disease}_ddpm_metrics.json"
    scores.to_csv(scores_path, index=False)
    metrics = summarize_scores(
        disease=disease,
        scores=scores,
        config=config,
        checkpoint_path=args.checkpoint,
        checkpoint=checkpoint,
    )
    coverage = evaluate_coverage_from_saved_samples(
        sample_paths=sample_paths,
        dataset=dataset,
        target_metadata=target_metadata,
        interval_levels=args.interval_levels,
    )
    coverage_output = {
        "checkpoint": str(args.checkpoint),
        "variant": "ddpm",
        "dataset_name": config["data"]["dataset_name"],
        "target_group": disease,
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
    metrics["coverage"] = coverage["aggregate"]
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    coverage_json_path = output_dir / f"{disease}_ddpm_coverage.json"
    coverage_csv_path = output_dir / f"{disease}_ddpm_coverage.csv"
    coverage_json_path.write_text(json.dumps(coverage_output, indent=2) + "\n")
    write_feature_interval_csv(coverage_csv_path, coverage_output)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"saved scores: {scores_path}", flush=True)
    print(f"saved metrics: {metrics_path}", flush=True)
    print(f"saved coverage: {coverage_json_path}", flush=True)
    print(f"saved coverage csv: {coverage_csv_path}", flush=True)


@torch.no_grad()
def sample_targets(
    *,
    diffusion: torch.nn.Module,
    source_encoder: torch.nn.Module | None,
    dataset: Any,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    nums = []
    cats = []
    n_rows = len(dataset)
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        source_num = dataset.source_num[start:end].to(device=device, dtype=torch.float32)
        source_cat = dataset.source_cat[start:end].to(device=device, dtype=torch.long)
        source_num_mask = dataset.source_num_mask[start:end].to(device=device, dtype=torch.float32)
        source_cat_mask = dataset.source_cat_mask[start:end].to(device=device, dtype=torch.float32)
        if source_encoder is None:
            sample = diffusion.sample(
                batch_size=end - start,
                model_kwargs={
                    "source_num": source_num,
                    "source_cat": source_cat,
                    "source_num_mask": source_num_mask,
                    "source_cat_mask": source_cat_mask,
                },
            )
        else:
            condition = source_encoder(source_num, source_cat, source_num_mask, source_cat_mask)
            sample = diffusion.sample(batch_size=end - start, condition=condition)
        nums.append(sample["num"].detach().cpu().numpy())
        cats.append(sample["cat"].detach().cpu().numpy())
    return {
        "num": np.concatenate(nums, axis=0) if nums else np.empty((0, 0), dtype=np.float32),
        "cat": np.concatenate(cats, axis=0) if cats else np.empty((0, 0), dtype=np.int64),
    }


def score_samples(
    *,
    disease: str,
    sample_paths: list[Path],
    source_context: dict[str, np.ndarray],
    labels: list[bool | None],
) -> pd.DataFrame:
    row_ids: np.ndarray | None = None
    positive = None
    observed = None
    for sample_path in sample_paths:
        sample = pd.read_csv(sample_path)
        current_row_ids = sample["row_id"].to_numpy(dtype=np.int64)
        if row_ids is None:
            row_ids = current_row_ids
            positive = np.zeros(len(row_ids), dtype=np.int64)
            observed = np.zeros(len(row_ids), dtype=np.int64)
        elif not np.array_equal(row_ids, current_row_ids):
            raise ValueError(f"row_id order differs in {sample_path}.")
        target = {
            column: sample[column].to_numpy()
            for column in sample.columns
            if column not in {"row_id", "sample"}
        }
        predictions = disease_predictions(disease, target, source_context)
        for idx, value in enumerate(predictions):
            if value is None:
                continue
            observed[idx] += 1
            positive[idx] += int(value)

    if row_ids is None or positive is None or observed is None:
        raise ValueError("No sample paths were provided.")
    probabilities = [
        None if count == 0 else pos / float(count)
        for pos, count in zip(positive.tolist(), observed.tolist())
    ]
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


def summarize_scores(
    *,
    disease: str,
    scores: pd.DataFrame,
    config: dict[str, Any],
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    probabilities = [
        None if pd.isna(value) else float(value)
        for value in scores["disease_probability"].tolist()
    ]
    labels = [
        None if pd.isna(value) else bool(value)
        for value in scores["y_true"].tolist()
    ]
    predictions = [
        None if value is None else bool(value >= 0.5)
        for value in probabilities
    ]
    return {
        "variant": "ddpm",
        "target_group": disease,
        "dataset_name": config["data"]["dataset_name"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "best_valid_loss": float(checkpoint.get("best_valid_loss", float("nan"))),
        "num_timesteps": int(config["diffusion"]["num_timesteps"]),
        "num_samples": int(scores["num_samples"].iloc[0]) if len(scores) else 0,
        "auroc": binary_auroc(probabilities, labels),
        **binary_metrics(predictions, labels),
    }


def evaluate_coverage_from_saved_samples(
    *,
    sample_paths: list[Path],
    dataset: Any,
    target_metadata: dict[str, Any],
    interval_levels: list[float],
) -> dict[str, Any]:
    num_features = list(target_metadata["features"]["num"])
    if not num_features:
        return {"features": [], "aggregate": {"n_features": 0, "n_observed": 0}}

    target_num = dataset.target_num.detach().cpu().numpy()
    target_mask = dataset.target_num_mask.detach().cpu().numpy() > 0
    states = target_metadata["gaussian_quantile"]["features"]
    accumulators = {
        feature: FeatureCoverageAccumulator(feature=feature, interval_levels=interval_levels)
        for feature in num_features
    }
    sample_frames = [pd.read_csv(path) for path in sample_paths]
    if not sample_frames:
        raise ValueError("No sample paths were provided.")

    row_ids = sample_frames[0]["row_id"].to_numpy(dtype=np.int64)
    for sample_path, frame in zip(sample_paths[1:], sample_frames[1:]):
        if not np.array_equal(row_ids, frame["row_id"].to_numpy(dtype=np.int64)):
            raise ValueError(f"row_id order differs in {sample_path}.")

    for feature_idx, feature in enumerate(num_features):
        observed = target_mask[:, feature_idx]
        if not np.any(observed):
            continue
        generated_raw = np.stack(
            [frame[feature].to_numpy(dtype=np.float64) for frame in sample_frames],
            axis=1,
        )[observed]
        observed_raw = restore_one_target_feature(
            target_values=target_num[observed, feature_idx],
            state=states[feature_idx],
        )
        accumulators[feature].update(generated_raw=generated_raw, observed_raw=observed_raw)

    features = [accumulators[feature].finalize() for feature in num_features]
    return {
        "features": features,
        "aggregate": aggregate_feature_results(features),
    }


def restore_one_target_feature(*, target_values: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    from eval.cvae_common import inverse_gaussian_quantile

    return inverse_gaussian_quantile(target_values, state)


def read_labels(
    config: dict[str, Any],
    disease: str,
    row_ids: np.ndarray,
) -> list[bool | None]:
    dataset_name = str(config["data"]["dataset_name"])
    suffix = dataset_name.removeprefix("harmonized_")
    path = Path("datasets/harmonized") / f"disease_labels_harmonized_{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    column = f"disease_{disease}"
    labels_by_index: dict[int, bool | None] = {}
    with path.open(newline="", errors="replace") as file:
        reader = csv.DictReader(file)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"{path} does not contain {column}.")
        for row in reader:
            labels_by_index[int(row["row_index"])] = parse_label(row[column])
    return [labels_by_index[int(idx)] for idx in row_ids]


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--interval-levels", nargs="+", type=float, default=[0.9, 0.95, 0.99])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite-samples", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
