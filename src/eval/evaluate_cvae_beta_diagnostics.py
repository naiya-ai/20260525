"""Evaluate CVAE beta-sweep diagnostics for calibration, prior reconstruction, and KL usage."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval.cvae_common import (
    DISEASES,
    binary_auroc,
    build_model,
    decode_source_context,
    inverse_gaussian_quantile,
    load_checkpoint_config_and_schema,
    read_target_metadata,
    resolve_device,
    set_seed,
)
from eval.evaluate_cvae_prior_probability import (
    decoded_sample_disease_probabilities,
    load_split_labels,
)
from train.dataset import create_grouped_cvae_dataloader, load_grouped_cvae_schema


def main() -> None:
    args = parse_args()
    config, _ = load_checkpoint_config_and_schema(args)
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
    model = build_model(config, schema)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    target_metadata = read_target_metadata(config)

    output = {
        "checkpoint": str(args.checkpoint),
        "variant": args.variant,
        "target_group": args.target_group,
        "dataset_name": config["data"]["dataset_name"],
        "num_prior_samples": args.num_samples,
        "coverage_levels": args.coverage_levels,
        "splits": {},
    }
    for split in args.splits:
        loader = create_grouped_cvae_dataloader(
            config,
            split=split,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        labels = load_split_labels(config, args.target_group, split=split)
        labels = labels[: args.max_rows] if args.max_rows is not None else labels
        supplemental_source_context = load_supplemental_source_context(
            config,
            target_metadata=target_metadata,
            split=split,
            max_rows=args.max_rows,
        )
        split_result = evaluate_split(
            model=model,
            loader=loader,
            labels=labels,
            schema=schema,
            config=config,
            target_metadata=target_metadata,
            disease=args.target_group,
            num_samples=args.num_samples,
            coverage_levels=args.coverage_levels,
            device=device,
            cache_prior_samples=should_cache_split(args, split),
            supplemental_source_context=supplemental_source_context,
        )
        cache = split_result.pop("_prior_sample_cache", None)
        if cache is not None:
            write_prior_sample_cache(
                args.prior_sample_cache_dir
                / f"{args.variant}_{args.target_group}_{split}_prior_samples.npz",
                cache,
                variant=args.variant,
                target_group=args.target_group,
                dataset_name=config["data"]["dataset_name"],
                split=split,
                num_samples=args.num_samples,
                seed=args.seed,
            )
        output["splits"][split] = split_result

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"{args.variant}_{args.target_group}_diagnostics.json"
    result_path.write_text(json.dumps(output, indent=2) + "\n")
    if args.save_probabilities:
        for split, split_result in output["splits"].items():
            write_probabilities(
                args.output_dir / f"{args.variant}_{args.target_group}_{split}_probabilities.csv",
                probabilities=split_result["probabilities"],
                labels=split_result["labels"],
            )
    print(json.dumps(output, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-group", required=True, choices=DISEASES)
    parser.add_argument("--variant", default="cvae_beta")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--dataset-name")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--splits", nargs="+", choices=["valid", "test"], default=["valid", "test"])
    parser.add_argument("--coverage-levels", nargs="+", type=float, default=[0.5, 0.8, 0.9, 0.95])
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--prior-sample-cache-dir", type=Path)
    parser.add_argument("--prior-sample-cache-splits", nargs="+", choices=["valid", "test"], default=["test"])
    return parser.parse_args()


@torch.no_grad()
def evaluate_split(
    *,
    model,
    loader,
    labels: list[bool | None],
    schema: dict[str, Any],
    config: dict[str, Any],
    target_metadata: dict[str, Any],
    disease: str,
    num_samples: int,
    coverage_levels: list[float],
    device: torch.device,
    cache_prior_samples: bool = False,
    supplemental_source_context: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    probabilities: list[float | None] = []
    sample_values: list[list[float]] = []
    labels_seen: list[bool | None] = []
    prior_reconstruction = ReconstructionAccumulator(
        target_metadata,
        num_prior_samples=num_samples,
    )
    latent_usage = LatentUsageAccumulator()
    cache_generated_num_parts: list[np.ndarray] = []
    cache_target_num_parts: list[np.ndarray] = []
    cache_target_num_mask_parts: list[np.ndarray] = []

    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        batch_size = int(batch["source_num"].shape[0])
        label_slice = labels[len(labels_seen) : len(labels_seen) + batch_size]
        labels_seen.extend(label_slice)

        outputs = model(
            source_num=batch["source_num"],
            source_cat=batch["source_cat"],
            source_num_mask=batch["source_num_mask"],
            source_cat_mask=batch["source_cat_mask"],
            target_num=batch["target_num"],
            target_cat=batch["target_cat"],
            target_num_mask=batch["target_num_mask"],
            target_cat_mask=batch["target_cat_mask"],
            sample_latent=False,
        )
        latent_usage.update(
            mu=outputs["mu"],
            logvar=outputs["logvar"],
            prior_mu=outputs["prior_mu"],
            prior_logvar=outputs["prior_logvar"],
        )

        condition = model.encode_condition(
            source_num=batch["source_num"],
            source_cat=batch["source_cat"],
            source_num_mask=batch["source_num_mask"],
            source_cat_mask=batch["source_cat_mask"],
        )
        source_context = decode_source_context(batch=batch, schema=schema, config=config)
        if supplemental_source_context:
            start = len(labels_seen) - batch_size
            stop = len(labels_seen)
            for key, values in supplemental_source_context.items():
                source_context.setdefault(key, values[start:stop])
        disease_samples = np.full((batch_size, num_samples), np.nan, dtype=np.float64)
        num_sum = torch.zeros_like(batch["target_num"])
        cat_prob_sum = None
        generated_num = (
            np.empty(
                (batch_size, num_samples, int(batch["target_num"].shape[1])),
                dtype=np.float32,
            )
            if cache_prior_samples
            else None
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
            sample_probabilities = decoded_sample_disease_probabilities(
                decoded=decoded,
                target_metadata=target_metadata,
                disease=disease,
                source_context=source_context,
            )
            for row_idx, probability in enumerate(sample_probabilities):
                if probability is not None:
                    disease_samples[row_idx, sample_idx] = float(probability)
            num_sum += decoded["num_mean"]
            if generated_num is not None:
                generated_num[:, sample_idx, :] = decoded["num_mean"].detach().float().cpu().numpy()
            if "cat_logits" in decoded:
                probs = torch.softmax(decoded["cat_logits"], dim=1)
                cat_prob_sum = probs if cat_prob_sum is None else cat_prob_sum + probs

        for row in disease_samples:
            observed = row[~np.isnan(row)]
            sample_values.append([float(value) for value in observed])
            probabilities.append(None if len(observed) == 0 else float(np.mean(observed)))
        prior_reconstruction.update(
            pred_num_mean=num_sum / float(num_samples),
            pred_cat_prob=None if cat_prob_sum is None else cat_prob_sum / float(num_samples),
            target_num=batch["target_num"],
            target_cat=batch["target_cat"],
            target_num_mask=batch["target_num_mask"],
            target_cat_mask=batch["target_cat_mask"],
        )
        if generated_num is not None:
            cache_generated_num_parts.append(generated_num)
            cache_target_num_parts.append(batch["target_num"].detach().float().cpu().numpy())
            cache_target_num_mask_parts.append(batch["target_num_mask"].detach().float().cpu().numpy())

    result = {
        "n_rows": len(probabilities),
        "labels": labels_seen,
        "probabilities": probabilities,
        "coverage_calibration": calibration_metrics(
            probabilities=probabilities,
            sample_values=sample_values,
            labels=labels_seen,
            coverage_levels=coverage_levels,
        ),
        "conditional_prior_reconstruction": prior_reconstruction.finalize(),
        "kl_latent_usage": latent_usage.finalize(),
    }
    if cache_prior_samples:
        result["_prior_sample_cache"] = {
            "generated_num": np.concatenate(cache_generated_num_parts, axis=0),
            "target_num": np.concatenate(cache_target_num_parts, axis=0),
            "target_num_mask": np.concatenate(cache_target_num_mask_parts, axis=0),
            "num_feature_names": np.asarray(target_metadata["features"]["num"], dtype=object),
        }
    return result


def load_supplemental_source_context(
    config: dict[str, Any],
    *,
    target_metadata: dict[str, Any],
    split: str,
    max_rows: int | None,
) -> dict[str, np.ndarray]:
    """Load rule-only context such as age/sex when it is not part of source groups.

    Compact source groups may intentionally omit demographics. Disease-rule AUROC
    still needs those values to label decoded target samples, so read them from the
    harmonized view in split order without changing the model input.
    """

    dataset_root = Path(config["data"]["dataset_root"])
    dataset_name = str(config["data"]["dataset_name"])
    target_group = str(config["data"]["target_group"])
    split_path = dataset_root / dataset_name / target_group / "split.csv"
    split_indices: list[int] = []
    with split_path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["split"] == split:
                split_indices.append(int(row["row_index"]))
    if max_rows is not None:
        split_indices = split_indices[: int(max_rows)]
    if not split_indices:
        return {}

    features = ("age", "sex")
    harmonized_path = Path(target_metadata["harmonized_dataset"])
    row_filter = target_metadata.get("harmonized_filter")
    wanted = set(split_indices)
    values_by_filtered_index: dict[int, dict[str, float]] = {}
    filtered_index = -1
    with harmonized_path.open(newline="", encoding="utf-8-sig", errors="replace") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if not harmonized_row_matches_filter(row, row_filter):
                continue
            filtered_index += 1
            if filtered_index not in wanted:
                continue
            values_by_filtered_index[filtered_index] = {
                feature: parse_optional_float(row.get(feature, ""))
                for feature in features
            }
            if len(values_by_filtered_index) == len(wanted):
                break

    context: dict[str, np.ndarray] = {}
    for feature in features:
        context[feature] = np.asarray(
            [
                values_by_filtered_index.get(index, {}).get(feature, np.nan)
                for index in split_indices
            ],
            dtype=np.float64,
        )
    return context


def harmonized_row_matches_filter(row: dict[str, str], row_filter: dict[str, Any] | None) -> bool:
    if not row_filter:
        return True
    surveys = row_filter.get("surveys")
    if surveys:
        survey = normalize_survey_name(row.get("survey", ""))
        if not survey:
            survey = normalize_survey_name(row.get("source", ""))
        if survey not in {normalize_survey_name(value) for value in surveys}:
            return False
    year = parse_optional_float(row.get("year", ""))
    year_min = row_filter.get("year_min")
    year_max = row_filter.get("year_max")
    if year_min is not None and (year is None or year < float(year_min)):
        return False
    if year_max is not None and (year is None or year > float(year_max)):
        return False
    return True


def normalize_survey_name(value: Any) -> str:
    key = str(value).strip().lower()
    aliases = {
        "1": "knhanes",
        "1.0": "knhanes",
        "knhanes": "knhanes",
        "khanes": "knhanes",
        "k": "knhanes",
        "2": "nhanes",
        "2.0": "nhanes",
        "nhanes": "nhanes",
        "n": "nhanes",
    }
    return aliases.get(key, key)


def parse_optional_float(value: Any) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return float("nan")


def should_cache_split(args: argparse.Namespace, split: str) -> bool:
    return args.prior_sample_cache_dir is not None and split in set(args.prior_sample_cache_splits)


def write_prior_sample_cache(
    path: Path,
    cache: dict[str, np.ndarray],
    *,
    variant: str,
    target_group: str,
    dataset_name: str,
    split: str,
    num_samples: int,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        generated_num=cache["generated_num"],
        target_num=cache["target_num"],
        target_num_mask=cache["target_num_mask"],
        num_feature_names=cache["num_feature_names"],
        variant=np.asarray(variant),
        target_group=np.asarray(target_group),
        dataset_name=np.asarray(dataset_name),
        split=np.asarray(split),
        num_samples=np.asarray(int(num_samples), dtype=np.int64),
        seed=np.asarray(int(seed), dtype=np.int64),
    )


class ReconstructionAccumulator:
    def __init__(self, target_metadata: dict[str, Any], *, num_prior_samples: int) -> None:
        self.target_metadata = target_metadata
        self.num_prior_samples = int(num_prior_samples)
        self.n_num = 0
        self.num_abs = None
        self.num_sq = None
        self.num_count = None
        self.raw_abs = None
        self.raw_sq = None
        self.raw_count = None
        self.cat_correct = 0
        self.cat_count = 0
        self.cat_nll_sum = 0.0

    def update(
        self,
        *,
        pred_num_mean: torch.Tensor,
        pred_cat_prob: torch.Tensor | None,
        target_num: torch.Tensor,
        target_cat: torch.Tensor,
        target_num_mask: torch.Tensor,
        target_cat_mask: torch.Tensor,
    ) -> None:
        pred_np = pred_num_mean.detach().cpu().numpy()
        target_np = target_num.detach().cpu().numpy()
        mask_np = target_num_mask.detach().cpu().numpy() > 0
        abs_err = np.abs(pred_np - target_np) * mask_np
        sq_err = np.square(pred_np - target_np) * mask_np
        counts = mask_np.sum(axis=0).astype(np.float64)
        self.num_abs = add_or_init(self.num_abs, abs_err.sum(axis=0))
        self.num_sq = add_or_init(self.num_sq, sq_err.sum(axis=0))
        self.num_count = add_or_init(self.num_count, counts)
        self.n_num += int(mask_np.sum())

        raw_abs = np.zeros((pred_np.shape[1],), dtype=np.float64)
        raw_sq = np.zeros((pred_np.shape[1],), dtype=np.float64)
        raw_count = np.zeros((pred_np.shape[1],), dtype=np.float64)
        states = self.target_metadata["gaussian_quantile"]["features"]
        for idx, state in enumerate(states):
            observed = mask_np[:, idx]
            if not np.any(observed):
                continue
            pred_raw = inverse_gaussian_quantile(pred_np[observed, idx], state)
            target_raw = inverse_gaussian_quantile(target_np[observed, idx], state)
            diff = pred_raw - target_raw
            raw_abs[idx] = float(np.abs(diff).sum())
            raw_sq[idx] = float(np.square(diff).sum())
            raw_count[idx] = float(observed.sum())
        self.raw_abs = add_or_init(self.raw_abs, raw_abs)
        self.raw_sq = add_or_init(self.raw_sq, raw_sq)
        self.raw_count = add_or_init(self.raw_count, raw_count)

        if pred_cat_prob is not None and target_cat.shape[1] > 0:
            pred_cat_np = pred_cat_prob.detach().cpu().numpy()
            target_cat_np = target_cat.detach().cpu().numpy()
            target_cat_mask_np = target_cat_mask.detach().cpu().numpy() > 0
            offset = 0
            for idx, size in enumerate(self.target_metadata["category_sizes"]):
                size = int(size)
                pred_idx = pred_cat_np[:, offset : offset + size].argmax(axis=1)
                observed = target_cat_mask_np[:, idx]
                self.cat_correct += int((pred_idx[observed] == target_cat_np[observed, idx]).sum())
                self.cat_count += int(observed.sum())
                if np.any(observed):
                    true_idx = target_cat_np[observed, idx].astype(np.int64)
                    true_prob = pred_cat_np[observed, offset : offset + size][
                        np.arange(int(observed.sum())),
                        true_idx,
                    ]
                    self.cat_nll_sum += float(-np.log(np.clip(true_prob, 1e-12, 1.0)).sum())
                offset += size

    def finalize(self) -> dict[str, Any]:
        num_features = list(self.target_metadata["features"]["num"])
        result = {
            "estimator": "prior_sample_mean",
            "num_prior_samples": self.num_prior_samples,
            "transformed_num_mae": safe_ratio(float(np.sum(self.num_abs)), float(np.sum(self.num_count))),
            "transformed_num_rmse": safe_sqrt_ratio(float(np.sum(self.num_sq)), float(np.sum(self.num_count))),
            "raw_num_mae": safe_ratio(float(np.sum(self.raw_abs)), float(np.sum(self.raw_count))),
            "raw_num_rmse": safe_sqrt_ratio(float(np.sum(self.raw_sq)), float(np.sum(self.raw_count))),
            "cat_accuracy": safe_ratio(self.cat_correct, self.cat_count),
            "cat_nll": safe_ratio(self.cat_nll_sum, self.cat_count),
            "cat_bce": safe_ratio(self.cat_nll_sum, self.cat_count),
            "per_numeric_feature": {},
        }
        for idx, feature in enumerate(num_features):
            result["per_numeric_feature"][feature] = {
                "estimator": "prior_sample_mean",
                "num_prior_samples": self.num_prior_samples,
                "transformed_mae": safe_ratio(float(self.num_abs[idx]), float(self.num_count[idx])),
                "transformed_rmse": safe_sqrt_ratio(float(self.num_sq[idx]), float(self.num_count[idx])),
                "raw_mae": safe_ratio(float(self.raw_abs[idx]), float(self.raw_count[idx])),
                "raw_rmse": safe_sqrt_ratio(float(self.raw_sq[idx]), float(self.raw_count[idx])),
                "n_observed": int(self.num_count[idx]),
            }
        return result


class LatentUsageAccumulator:
    def __init__(self) -> None:
        self.n = 0
        self.kl_sum = None
        self.kl_sq_sum = None
        self.mu_sum = None
        self.mu_sq_sum = None
        self.prior_mu_sum = None
        self.prior_mu_sq_sum = None

    def update(
        self,
        *,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        prior_mu: torch.Tensor,
        prior_logvar: torch.Tensor,
    ) -> None:
        kl_dim = 0.5 * (
            prior_logvar
            - logvar
            + torch.exp(logvar - prior_logvar)
            + torch.square(mu - prior_mu) * torch.exp(-prior_logvar)
            - 1.0
        )
        kl_np = kl_dim.detach().cpu().numpy().astype(np.float64)
        mu_np = mu.detach().cpu().numpy().astype(np.float64)
        prior_mu_np = prior_mu.detach().cpu().numpy().astype(np.float64)
        self.n += int(mu_np.shape[0])
        self.kl_sum = add_or_init(self.kl_sum, kl_np.sum(axis=0))
        self.kl_sq_sum = add_or_init(self.kl_sq_sum, np.square(kl_np).sum(axis=0))
        self.mu_sum = add_or_init(self.mu_sum, mu_np.sum(axis=0))
        self.mu_sq_sum = add_or_init(self.mu_sq_sum, np.square(mu_np).sum(axis=0))
        self.prior_mu_sum = add_or_init(self.prior_mu_sum, prior_mu_np.sum(axis=0))
        self.prior_mu_sq_sum = add_or_init(self.prior_mu_sq_sum, np.square(prior_mu_np).sum(axis=0))

    def finalize(self) -> dict[str, Any]:
        if self.n == 0:
            return {}
        kl_mean_dim = self.kl_sum / float(self.n)
        kl_var_dim = np.maximum(self.kl_sq_sum / float(self.n) - np.square(kl_mean_dim), 0.0)
        mu_mean = self.mu_sum / float(self.n)
        mu_var = np.maximum(self.mu_sq_sum / float(self.n) - np.square(mu_mean), 0.0)
        prior_mu_mean = self.prior_mu_sum / float(self.n)
        prior_mu_var = np.maximum(self.prior_mu_sq_sum / float(self.n) - np.square(prior_mu_mean), 0.0)
        return {
            "n_rows": self.n,
            "latent_dim": int(len(kl_mean_dim)),
            "kl_mean": float(np.sum(kl_mean_dim)),
            "kl_std_approx": float(np.sqrt(np.sum(kl_var_dim))),
            "kl_per_dim_mean": [float(value) for value in kl_mean_dim],
            "kl_per_dim_std": [float(value) for value in np.sqrt(kl_var_dim)],
            "active_units_kl_gt_0_001": int(np.sum(kl_mean_dim > 0.001)),
            "active_units_kl_gt_0_01": int(np.sum(kl_mean_dim > 0.01)),
            "posterior_mu_var_per_dim": [float(value) for value in mu_var],
            "prior_mu_var_per_dim": [float(value) for value in prior_mu_var],
            "active_units_posterior_mu_var_gt_0_01": int(np.sum(mu_var > 0.01)),
        }


def calibration_metrics(
    *,
    probabilities: list[float | None],
    sample_values: list[list[float]],
    labels: list[bool | None],
    coverage_levels: list[float],
) -> dict[str, Any]:
    observed = [
        (float(probability), bool(label), values)
        for probability, label, values in zip(probabilities, labels, sample_values)
        if probability is not None and label is not None
    ]
    if not observed:
        return {
            "n_evaluable": 0,
            "auroc": None,
            "brier_score": None,
            "log_loss": None,
            "ece_10": None,
            "coverage": {},
        }

    probs = np.asarray([item[0] for item in observed], dtype=np.float64)
    y = np.asarray([item[1] for item in observed], dtype=np.float64)
    clipped = np.clip(probs, 1e-7, 1.0 - 1e-7)
    coverage = {}
    for level in coverage_levels:
        alpha = 1.0 - float(level)
        hits = []
        widths = []
        for _, label, values in observed:
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            lower = float(np.quantile(arr, alpha / 2.0))
            upper = float(np.quantile(arr, 1.0 - alpha / 2.0))
            y_value = 1.0 if label else 0.0
            hits.append(lower <= y_value <= upper)
            widths.append(upper - lower)
        coverage[f"{level:.2f}"] = {
            "empirical_coverage": none_if_empty_mean(hits),
            "mean_interval_width": none_if_empty_mean(widths),
        }

    return {
        "n_evaluable": int(len(observed)),
        "prevalence": float(y.mean()),
        "mean_predicted_probability": float(probs.mean()),
        "auroc": binary_auroc(probabilities, labels),
        "brier_score": float(np.mean(np.square(probs - y))),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))),
        "ece_10": expected_calibration_error(probs, y, n_bins=10),
        "calibration_bins_10": calibration_bins(probs, y, n_bins=10),
        "coverage": coverage,
    }


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, *, n_bins: int) -> float:
    total = len(probs)
    error = 0.0
    for bin_result in calibration_bins(probs, y, n_bins=n_bins):
        count = int(bin_result["count"])
        if count == 0:
            continue
        error += (count / total) * abs(bin_result["mean_probability"] - bin_result["event_rate"])
    return float(error)


def calibration_bins(probs: np.ndarray, y: np.ndarray, *, n_bins: int) -> list[dict[str, Any]]:
    bins = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for idx in range(n_bins):
        lower = edges[idx]
        upper = edges[idx + 1]
        if idx == n_bins - 1:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)
        count = int(mask.sum())
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_probability": None if count == 0 else float(probs[mask].mean()),
                "event_rate": None if count == 0 else float(y[mask].mean()),
            }
        )
    return bins


def write_probabilities(
    path: Path,
    *,
    probabilities: list[float | None],
    labels: list[bool | None],
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["index", "disease_probability", "label"])
        writer.writeheader()
        for idx, (probability, label) in enumerate(zip(probabilities, labels)):
            writer.writerow({"index": idx, "disease_probability": probability, "label": label})


def add_or_init(current: np.ndarray | None, value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if current is None:
        return value.copy()
    return current + value


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def safe_sqrt_ratio(numerator: float, denominator: float) -> float | None:
    ratio = safe_ratio(numerator, denominator)
    if ratio is None:
        return None
    return float(math.sqrt(ratio))


def none_if_empty_mean(values: list[float] | list[bool]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


if __name__ == "__main__":
    main()
