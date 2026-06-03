"""Evaluate saved CatBoost disease models with validation-selected thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from catboost import CatBoostClassifier, Pool

from baselines.catboost_disease import (
    DISEASES,
    binary_metrics,
    label_counts,
    load_source_frame,
    observed_indices_and_labels,
    read_disease_labels,
    read_split,
)
from eval.cvae_common import binary_auroc
from eval.evaluate_cvae_prior_probability import probabilities_to_predictions, select_threshold


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_root / args.dataset_name
    split_by_name = read_split(dataset_dir / args.target_group / "split.csv")
    labels = read_disease_labels(args.dataset_name, args.target_group)
    source = load_source_frame(dataset_dir, args.source_groups)
    cat_features = [column for column in source.columns if column.startswith("cat:")]

    valid_indices, valid_y = observed_indices_and_labels(split_by_name["valid"], labels)
    test_indices = split_by_name["test"]
    test_labels = [labels.get(idx) for idx in test_indices]

    model = CatBoostClassifier()
    model.load_model(args.model)

    valid_pool = Pool(source.iloc[valid_indices], cat_features=cat_features)
    test_pool = Pool(source.iloc[test_indices], cat_features=cat_features)
    valid_probabilities = model.predict_proba(valid_pool)[:, 1].astype(float).tolist()
    valid_labels = [bool(value) for value in valid_y.tolist()]
    test_probabilities = model.predict_proba(test_pool)[:, 1].astype(float).tolist()

    threshold_result = select_threshold(
        probabilities=valid_probabilities,
        labels=valid_labels,
        strategy="validation_balanced_accuracy",
        fixed_threshold=args.default_threshold,
    )
    test_predictions = probabilities_to_predictions(
        test_probabilities,
        threshold=threshold_result["threshold"],
    )
    metrics = binary_metrics(test_predictions, test_labels)
    auroc = binary_auroc(test_probabilities, test_labels)

    result: dict[str, Any] = {
        "variant": args.variant,
        "target_group": args.target_group,
        "dataset_name": args.dataset_name,
        "model": str(args.model),
        "threshold_strategy": "validation_balanced_accuracy",
        "threshold": threshold_result["threshold"],
        "validation_threshold_metrics": threshold_result["metrics"],
        "valid_label_counts": label_counts(valid_y),
        "test_observed_label_counts": label_counts(
            np.asarray([int(label) for label in test_labels if label is not None], dtype=np.int64)
        ),
        "auroc": auroc,
        **metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"{args.variant}_{args.target_group}_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/preprocessed/gaussian_quantile"))
    parser.add_argument("--dataset-name", default="harmonized_knhanes_1998_2024")
    parser.add_argument(
        "--source-groups",
        nargs="+",
        default=["questionnaire_without_disease", "dietary"],
    )
    parser.add_argument("--target-group", required=True, choices=DISEASES)
    parser.add_argument("--variant", default="catboost")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--default-threshold", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    main()
