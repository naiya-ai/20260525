"""Evaluate raw-scale numerical prediction interval coverage for CVAE priors."""

from __future__ import annotations

import argparse
import csv
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval.cvae_common import (
    DISEASES,
    build_model,
    inverse_gaussian_quantile,
    load_checkpoint_config_and_schema,
    read_target_metadata,
    resolve_device,
    set_seed,
)
from train.dataset import create_grouped_cvae_dataloader, load_grouped_cvae_schema


def main() -> None:
    args = parse_args()
    config, _ = load_checkpoint_config_and_schema(args)
    target_group = args.target_group or str(config["data"]["target_group"])
    config["data"]["target_group"] = target_group
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
    used_sample_cache = False
    if args.prior_sample_cache is not None and args.prior_sample_cache.exists():
        result = evaluate_coverage_from_cache(
            cache_path=args.prior_sample_cache,
            target_metadata=target_metadata,
            num_samples=args.num_samples,
            interval_levels=args.interval_levels,
        )
        used_sample_cache = True
    else:
        model = build_model(config, schema)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device).eval()
        loader = create_grouped_cvae_dataloader(
            config,
            split=args.split,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        result = evaluate_coverage(
            model=model,
            loader=loader,
            target_metadata=target_metadata,
            num_samples=args.num_samples,
            interval_levels=args.interval_levels,
            device=device,
            mixed_precision=args.mixed_precision,
        )
    output = {
        "checkpoint": str(args.checkpoint),
        "variant": args.variant,
        "dataset_name": config["data"]["dataset_name"],
        "target_group": target_group,
        "source_groups": list(config["data"]["source_groups"]),
        "split": args.split,
        "num_prior_samples": args.num_samples,
        "interval_levels": args.interval_levels,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "mixed_precision": bool(args.mixed_precision),
        "prior_sample_cache": None if args.prior_sample_cache is None else str(args.prior_sample_cache),
        "used_sample_cache": used_sample_cache,
        **result,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.variant}_{target_group}_{args.split}_coverage.json"
    json_path.write_text(json.dumps(output, indent=2) + "\n")
    if args.save_csv:
        write_feature_interval_csv(
            args.output_dir / f"{args.variant}_{target_group}_{args.split}_coverage.csv",
            output,
        )
    print(json.dumps(output, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-group", choices=DISEASES)
    parser.add_argument("--variant", default="cvae")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--dataset-name")
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--interval-levels", nargs="+", type=float, default=[0.9, 0.95, 0.99])
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--prior-sample-cache", type=Path)
    return parser.parse_args()


def evaluate_coverage_from_cache(
    *,
    cache_path: Path,
    target_metadata: dict[str, Any],
    num_samples: int,
    interval_levels: list[float],
) -> dict[str, Any]:
    cache = np.load(cache_path, allow_pickle=True)
    generated = cache["generated_num"].astype(np.float32, copy=False)
    target_num = cache["target_num"].astype(np.float32, copy=False)
    target_mask = cache["target_num_mask"].astype(np.float32, copy=False) > 0
    cached_samples = int(np.asarray(cache["num_samples"]).item())
    if cached_samples != int(num_samples):
        raise ValueError(
            f"Prior sample cache has {cached_samples} samples, but --num-samples={num_samples}. "
            "Use matching DIAGNOSTICS_SAMPLES and COVERAGE_SAMPLES."
        )
    if generated.ndim != 3:
        raise ValueError(f"Expected generated_num shape [rows, samples, features], got {generated.shape}.")
    if generated.shape[0] != target_num.shape[0] or generated.shape[2] != target_num.shape[1]:
        raise ValueError("Prior sample cache generated_num and target_num shapes do not match.")

    num_features = list(target_metadata["features"]["num"])
    accumulators = {
        feature: FeatureCoverageAccumulator(feature=feature, interval_levels=interval_levels)
        for feature in num_features
    }
    states = target_metadata["gaussian_quantile"]["features"]
    for feature_idx, feature in enumerate(num_features):
        observed = target_mask[:, feature_idx]
        if not np.any(observed):
            continue
        state = states[feature_idx]
        generated_raw = inverse_gaussian_quantile(
            generated[observed, :, feature_idx].reshape(-1),
            state,
        ).reshape(int(observed.sum()), num_samples)
        observed_raw = inverse_gaussian_quantile(target_num[observed, feature_idx], state)
        accumulators[feature].update(generated_raw=generated_raw, observed_raw=observed_raw)

    features = [accumulators[feature].finalize() for feature in num_features]
    return {
        "features": features,
        "aggregate": aggregate_feature_results(features),
    }


@torch.no_grad()
def evaluate_coverage(
    *,
    model,
    loader,
    target_metadata: dict[str, Any],
    num_samples: int,
    interval_levels: list[float],
    device: torch.device,
    mixed_precision: bool,
) -> dict[str, Any]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    num_features = list(target_metadata["features"]["num"])
    if not num_features:
        return {"features": [], "aggregate": {"n_features": 0, "n_observed": 0}}

    accumulators = {
        feature: FeatureCoverageAccumulator(feature=feature, interval_levels=interval_levels)
        for feature in num_features
    }
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if mixed_precision and device.type == "cuda"
        else nullcontext()
    )

    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        batch_size = int(batch["source_num"].shape[0])
        n_num_features = int(batch["target_num"].shape[1])
        if n_num_features != len(num_features):
            raise ValueError(
                f"Metadata has {len(num_features)} numerical features but batch has {n_num_features}."
            )
        generated = np.empty((batch_size, num_samples, n_num_features), dtype=np.float32)
        with autocast_context:
            condition = model.encode_condition(
                source_num=batch["source_num"],
                source_cat=batch["source_cat"],
                source_num_mask=batch["source_num_mask"],
                source_cat_mask=batch["source_cat_mask"],
            )
            for sample_idx in range(num_samples):
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
                generated[:, sample_idx, :] = decoded["num_mean"].detach().float().cpu().numpy()

        target_num = batch["target_num"].detach().float().cpu().numpy()
        target_mask = batch["target_num_mask"].detach().float().cpu().numpy() > 0
        states = target_metadata["gaussian_quantile"]["features"]
        for feature_idx, feature in enumerate(num_features):
            observed = target_mask[:, feature_idx]
            if not np.any(observed):
                continue
            state = states[feature_idx]
            generated_raw = inverse_gaussian_quantile(
                generated[observed, :, feature_idx].reshape(-1),
                state,
            ).reshape(int(observed.sum()), num_samples)
            observed_raw = inverse_gaussian_quantile(target_num[observed, feature_idx], state)
            accumulators[feature].update(generated_raw=generated_raw, observed_raw=observed_raw)

    features = [accumulators[feature].finalize() for feature in num_features]
    return {
        "features": features,
        "aggregate": aggregate_feature_results(features),
    }


class FeatureCoverageAccumulator:
    def __init__(self, *, feature: str, interval_levels: list[float]) -> None:
        self.feature = feature
        self.interval_levels = interval_levels
        self.n_observed = 0
        self.generated_mean_sum = 0.0
        self.observed_sum = 0.0
        self.interval_hits = {level: 0 for level in interval_levels}
        self.interval_widths = {level: [] for level in interval_levels}

    def update(self, *, generated_raw: np.ndarray, observed_raw: np.ndarray) -> None:
        if generated_raw.ndim != 2:
            raise ValueError("generated_raw must have shape [rows, samples].")
        if generated_raw.shape[0] != observed_raw.shape[0]:
            raise ValueError("generated and observed row counts differ.")
        row_count = int(observed_raw.shape[0])
        self.n_observed += row_count
        self.generated_mean_sum += float(generated_raw.mean(axis=1).sum())
        self.observed_sum += float(observed_raw.sum())
        for level in self.interval_levels:
            lower_q = (1.0 - float(level)) / 2.0
            upper_q = 1.0 - lower_q
            lower = np.quantile(generated_raw, lower_q, axis=1)
            upper = np.quantile(generated_raw, upper_q, axis=1)
            self.interval_hits[level] += int(((lower <= observed_raw) & (observed_raw <= upper)).sum())
            self.interval_widths[level].extend((upper - lower).astype(np.float64).tolist())

    def finalize(self) -> dict[str, Any]:
        mean_generated = safe_divide(self.generated_mean_sum, self.n_observed)
        mean_observed = safe_divide(self.observed_sum, self.n_observed)
        intervals = []
        for level in self.interval_levels:
            widths = np.asarray(self.interval_widths[level], dtype=np.float64)
            coverage = safe_divide(self.interval_hits[level], self.n_observed)
            intervals.append(
                {
                    "interval_level": float(level),
                    "n_observed": self.n_observed,
                    "coverage": coverage,
                    "coverage_error": None if coverage is None else coverage - float(level),
                    "mean_interval_width": None if widths.size == 0 else float(widths.mean()),
                    "median_interval_width": None if widths.size == 0 else float(np.median(widths)),
                    "mean_generated": mean_generated,
                    "mean_observed": mean_observed,
                    "mean_bias": None
                    if mean_generated is None or mean_observed is None
                    else mean_generated - mean_observed,
                }
            )
        return {
            "feature": self.feature,
            "n_observed": self.n_observed,
            "intervals": intervals,
        }


def aggregate_feature_results(features: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        interval
        for feature in features
        for interval in feature["intervals"]
        if interval["coverage_error"] is not None
    ]
    if not rows:
        return {"n_features": len(features), "n_observed": 0}
    by_level: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        by_level.setdefault(float(row["interval_level"]), []).append(row)
    return {
        "n_features": len(features),
        "n_observed": int(sum(feature["n_observed"] for feature in features)),
        "by_interval_level": [
            {
                "interval_level": level,
                "mean_abs_coverage_error": float(
                    np.mean([abs(row["coverage_error"]) for row in level_rows])
                ),
                "mean_coverage_error": float(np.mean([row["coverage_error"] for row in level_rows])),
                "mean_interval_width": float(
                    np.mean([row["mean_interval_width"] for row in level_rows])
                ),
                "mean_abs_bias": float(np.mean([abs(row["mean_bias"]) for row in level_rows])),
            }
            for level, level_rows in sorted(by_level.items())
        ],
    }


def write_feature_interval_csv(path: Path, output: dict[str, Any]) -> None:
    fieldnames = [
        "checkpoint",
        "variant",
        "dataset_name",
        "target_group",
        "split",
        "source_groups",
        "num_prior_samples",
        "seed",
        "batch_size",
        "mixed_precision",
        "used_sample_cache",
        "prior_sample_cache",
        "feature",
        "n_observed",
        "interval_level",
        "coverage",
        "coverage_error",
        "mean_interval_width",
        "median_interval_width",
        "mean_generated",
        "mean_observed",
        "mean_bias",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for feature in output["features"]:
            for interval in feature["intervals"]:
                writer.writerow(
                    {
                        "checkpoint": output["checkpoint"],
                        "variant": output["variant"],
                        "dataset_name": output["dataset_name"],
                        "target_group": output["target_group"],
                        "split": output["split"],
                        "source_groups": " ".join(output["source_groups"]),
                        "num_prior_samples": output["num_prior_samples"],
                        "seed": output["seed"],
                        "batch_size": output["batch_size"],
                        "mixed_precision": output["mixed_precision"],
                        "used_sample_cache": output["used_sample_cache"],
                        "prior_sample_cache": output["prior_sample_cache"],
                        "feature": feature["feature"],
                        **interval,
                    }
                )


def safe_divide(numerator: float, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


if __name__ == "__main__":
    main()
