"""Evaluate CVAE disease metrics from multiple prior target samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval.cvae_common import (
    DISEASES,
    binary_auroc,
    binary_metrics,
    build_model,
    decode_source_context,
    disease_predictions,
    inverse_gaussian_quantile,
    load_checkpoint_config_and_schema,
    parse_label,
    read_target_metadata,
    resolve_device,
    set_seed,
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

    device = resolve_device(args.device or str(config.get("device", "cuda")))
    set_seed(args.seed)
    schema = load_grouped_cvae_schema(config)
    model = build_model(config, schema)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    target_metadata = read_target_metadata(config)

    valid_loader = create_grouped_cvae_dataloader(
        config,
        split="valid",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    test_loader = create_grouped_cvae_dataloader(
        config,
        split="test",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    valid_labels = load_split_labels(config, args.target_group, split="valid")
    test_labels = load_split_labels(config, args.target_group, split="test")

    valid_probabilities = prior_disease_probabilities(
        model=model,
        loader=valid_loader,
        schema=schema,
        config=config,
        target_metadata=target_metadata,
        disease=args.target_group,
        num_samples=args.num_samples,
        device=device,
    )
    threshold_result = select_threshold(
        probabilities=valid_probabilities,
        labels=valid_labels,
        strategy=args.threshold_strategy,
        fixed_threshold=args.threshold,
    )

    test_probabilities = prior_disease_probabilities(
        model=model,
        loader=test_loader,
        schema=schema,
        config=config,
        target_metadata=target_metadata,
        disease=args.target_group,
        num_samples=args.num_samples,
        device=device,
    )
    test_predictions = probabilities_to_predictions(
        test_probabilities,
        threshold=threshold_result["threshold"],
    )
    test_metrics = binary_metrics(test_predictions, test_labels)
    test_auroc = binary_auroc(test_probabilities, test_labels)

    result = {
        "target_group": args.target_group,
        "checkpoint": str(args.checkpoint),
        "variant": args.variant,
        "num_target_samples": args.num_samples,
        "threshold_strategy": args.threshold_strategy,
        "threshold": threshold_result["threshold"],
        "validation_threshold_metrics": threshold_result["metrics"],
        "n_predictions": len(test_predictions),
        "auroc": test_auroc,
        **test_metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"{args.variant}_{args.target_group}_metrics.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    if args.save_probabilities:
        write_probabilities(
            args.output_dir / f"{args.variant}_{args.target_group}_probabilities.csv",
            probabilities=test_probabilities,
            predictions=test_predictions,
            labels=test_labels,
        )
    print(json.dumps(result, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-group", required=True, choices=DISEASES)
    parser.add_argument("--variant", default="model_prior_probability")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--dataset-name")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument(
        "--threshold-strategy",
        choices=["fixed", "validation_balanced_accuracy"],
        default="validation_balanced_accuracy",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-probabilities", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def prior_disease_probabilities(
    *,
    model,
    loader,
    schema: dict[str, Any],
    config: dict[str, Any],
    target_metadata: dict[str, Any],
    disease: str,
    num_samples: int,
    device: torch.device,
) -> list[float | None]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    probabilities: list[float | None] = []
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        batch_size = int(batch["source_num"].shape[0])
        condition = model.encode_condition(
            source_num=batch["source_num"],
            source_cat=batch["source_cat"],
            source_num_mask=batch["source_num_mask"],
            source_cat_mask=batch["source_cat_mask"],
        )
        source_context = decode_source_context(batch=batch, schema=schema, config=config)
        positive_counts = np.zeros((batch_size,), dtype=np.float64)
        valid_counts = np.zeros((batch_size,), dtype=np.float64)
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
            sample_probabilities = decoded_sample_disease_probabilities(
                decoded=decoded,
                target_metadata=target_metadata,
                disease=disease,
                source_context=source_context,
            )
            for idx, probability in enumerate(sample_probabilities):
                if probability is None:
                    continue
                valid_counts[idx] += 1
                positive_counts[idx] += float(probability)
        for positive_count, valid_count in zip(positive_counts, valid_counts):
            if valid_count == 0:
                probabilities.append(None)
            else:
                probabilities.append(float(positive_count / valid_count))
    return probabilities


def decoded_sample_disease_probabilities(
    *,
    decoded: dict[str, torch.Tensor],
    target_metadata: dict[str, Any],
    disease: str,
    source_context: dict[str, np.ndarray],
) -> list[float | None]:
    if disease in {"hepatitis_b", "hepatitis_c"}:
        feature = "HE_hepaB" if disease == "hepatitis_b" else "HE_hepaC"
        return categorical_positive_probabilities(
            decoded=decoded,
            target_metadata=target_metadata,
            feature=feature,
            positive_category=2,
        )

    indicators = decode_generated_target_indicators(
        decoded=decoded,
        target_metadata=target_metadata,
    )
    predictions = disease_predictions(disease, indicators, source_context)
    return [None if prediction is None else float(prediction) for prediction in predictions]


def categorical_positive_probabilities(
    *,
    decoded: dict[str, torch.Tensor],
    target_metadata: dict[str, Any],
    feature: str,
    positive_category: int,
) -> list[float | None]:
    cat_features = list(target_metadata["features"]["cat"])
    if feature not in cat_features:
        raise ValueError(f"Target metadata does not contain categorical feature {feature!r}.")

    offset = 0
    for idx, candidate in enumerate(cat_features):
        size = int(target_metadata["category_sizes"][idx])
        if candidate == feature:
            if positive_category >= size:
                raise ValueError(
                    f"Positive category {positive_category} is out of range for {feature!r} "
                    f"with category size {size}."
                )
            logits = decoded["cat_logits"][:, offset : offset + size]
            probabilities = torch.softmax(logits, dim=1)[:, positive_category]
            return [float(value) for value in probabilities.detach().cpu().numpy()]
        offset += size
    raise AssertionError("unreachable")


def decode_generated_target_indicators(
    *,
    decoded: dict[str, torch.Tensor],
    target_metadata: dict[str, Any],
) -> dict[str, np.ndarray]:
    indicators: dict[str, np.ndarray] = {}
    num_mean = decoded["num_mean"].detach().cpu().numpy()
    for idx, feature in enumerate(target_metadata["features"]["num"]):
        state = target_metadata["gaussian_quantile"]["features"][idx]
        indicators[feature] = inverse_gaussian_quantile(num_mean[:, idx], state)

    cat_logits = decoded["cat_logits"]
    offset = 0
    for idx, feature in enumerate(target_metadata["features"]["cat"]):
        size = int(target_metadata["category_sizes"][idx])
        logits = cat_logits[:, offset : offset + size]
        indicators[feature] = logits.argmax(dim=1).detach().cpu().numpy()
        offset += size
    return indicators


def select_threshold(
    *,
    probabilities: list[float | None],
    labels: list[bool | None],
    strategy: str,
    fixed_threshold: float,
) -> dict[str, Any]:
    if strategy == "fixed":
        threshold = float(fixed_threshold)
        predictions = probabilities_to_predictions(probabilities, threshold=threshold)
        return {
            "threshold": threshold,
            "metrics": binary_metrics(predictions, labels),
        }
    if strategy != "validation_balanced_accuracy":
        raise ValueError(f"Unsupported threshold strategy: {strategy}")

    valid_probabilities = [
        probability
        for probability, label in zip(probabilities, labels)
        if probability is not None and label is not None
    ]
    if not valid_probabilities:
        threshold = float(fixed_threshold)
        predictions = probabilities_to_predictions(probabilities, threshold=threshold)
        return {
            "threshold": threshold,
            "metrics": binary_metrics(predictions, labels),
        }

    candidates = sorted(set([0.0, 1.0, *valid_probabilities]))
    best_threshold = float(fixed_threshold)
    best_score = -float("inf")
    best_metrics: dict[str, Any] | None = None
    for threshold in candidates:
        predictions = probabilities_to_predictions(probabilities, threshold=threshold)
        metrics = binary_metrics(predictions, labels)
        sensitivity = metrics["sensitivity"]
        specificity = metrics["specificity"]
        if sensitivity is None or specificity is None:
            continue
        score = 0.5 * (sensitivity + specificity)
        if score > best_score or (
            score == best_score and abs(threshold - fixed_threshold) < abs(best_threshold - fixed_threshold)
        ):
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics
    if best_metrics is None:
        predictions = probabilities_to_predictions(probabilities, threshold=best_threshold)
        best_metrics = binary_metrics(predictions, labels)
    return {
        "threshold": best_threshold,
        "metrics": best_metrics,
    }


def probabilities_to_predictions(
    probabilities: list[float | None],
    *,
    threshold: float,
) -> list[bool | None]:
    return [
        None if probability is None else bool(probability >= threshold)
        for probability in probabilities
    ]


def load_split_labels(config: dict[str, Any], disease: str, *, split: str) -> list[bool | None]:
    dataset_name = str(config["data"]["dataset_name"])
    suffix = dataset_name.removeprefix("harmonized_")
    path = Path("datasets/harmonized") / f"disease_labels_harmonized_{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    split_path = Path(config["data"]["dataset_root"]) / dataset_name / disease / "split.csv"
    split_indices: list[int] = []
    with split_path.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["split"] == split:
                split_indices.append(int(row["row_index"]))

    column = f"disease_{disease}"
    labels_by_index: dict[int, bool | None] = {}
    with path.open(newline="", errors="replace") as file:
        reader = csv.DictReader(file)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"{path} does not contain {column}.")
        for row in reader:
            labels_by_index[int(row["row_index"])] = parse_label(row[column])
    return [labels_by_index[idx] for idx in split_indices]


def write_probabilities(
    path: Path,
    *,
    probabilities: list[float | None],
    predictions: list[bool | None],
    labels: list[bool | None],
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["test_index", "disease_probability", "prediction", "label"],
        )
        writer.writeheader()
        for idx, (probability, prediction, label) in enumerate(
            zip(probabilities, predictions, labels)
        ):
            writer.writerow(
                {
                    "test_index": idx,
                    "disease_probability": probability,
                    "prediction": prediction,
                    "label": label,
                }
            )


if __name__ == "__main__":
    main()
