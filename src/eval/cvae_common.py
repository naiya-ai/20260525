"""Shared CVAE evaluation utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models import ConditionalMixedTypeVAE, build_conditional_vae_model
from train.dataset import load_grouped_cvae_schema


DISEASES = (
    "diabetes",
    "hypertension",
    "dyslipidemia",
    "liver_disease",
    "hepatitis_b",
    "hepatitis_c",
    "kidney_disease",
    "anemia",
)


def load_checkpoint_config_and_schema(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if config is None:
        raise ValueError(f"Checkpoint does not contain config: {args.checkpoint}")
    schema = checkpoint.get("schema")
    if schema is None:
        schema = load_grouped_cvae_schema(config)
    return config, schema


def build_model(config: dict[str, Any], schema: dict[str, Any]) -> ConditionalMixedTypeVAE:
    model_config = config["model"]
    return build_conditional_vae_model(
        source_n_num_features=schema["source"]["n_num_features"],
        source_category_sizes=schema["source"]["category_sizes"],
        target_n_num_features=schema["target"]["n_num_features"],
        target_category_sizes=schema["target"]["category_sizes"],
        model_config=model_config,
    )


def read_target_metadata(config: dict[str, Any]) -> dict[str, Any]:
    target_dir = (
        Path(config["data"]["dataset_root"])
        / str(config["data"]["dataset_name"])
        / str(config["data"]["target_group"])
    )
    return json.loads((target_dir / "metadata.json").read_text())


def decode_target_indicators(
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


def decode_source_context(
    *,
    batch: dict[str, torch.Tensor],
    schema: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    context: dict[str, np.ndarray] = {}
    source_schema = schema["source"]
    source_num_features = source_schema["numerical_features"]
    source_cat_features = source_schema["categorical_features"]
    source_num = batch["source_num"].detach().cpu().numpy()
    source_cat = batch["source_cat"].detach().cpu().numpy()
    source_num_mask = batch["source_num_mask"].detach().cpu().numpy()
    source_cat_mask = batch["source_cat_mask"].detach().cpu().numpy()
    source_metadata = read_source_metadata(config)
    quantile_by_feature = {}
    for metadata in source_metadata:
        for state in metadata["gaussian_quantile"]["features"]:
            quantile_by_feature[state["feature"]] = state

    if "age" in source_num_features:
        idx = source_num_features.index("age")
        age = inverse_gaussian_quantile(source_num[:, idx], quantile_by_feature["age"])
        age[source_num_mask[:, idx] <= 0] = np.nan
        context["age"] = age
    if "sex" in source_cat_features:
        idx = source_cat_features.index("sex")
        sex = source_cat[:, idx].astype(np.float64)
        sex[source_cat_mask[:, idx] <= 0] = np.nan
        context["sex"] = sex
    return context


def inverse_gaussian_quantile(values: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    if state.get("constant") or not state.get("quantiles"):
        quantiles = state.get("quantiles") or [0.0]
        return np.full_like(values, float(quantiles[0]), dtype=np.float64)
    normal_values = np.asarray(state["normal_values"], dtype=np.float64)
    quantiles = np.asarray(state["quantiles"], dtype=np.float64)
    return np.interp(values.astype(np.float64), normal_values, quantiles)


def read_source_metadata(config: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_dir = Path(config["data"]["dataset_root"]) / str(config["data"]["dataset_name"])
    return [
        json.loads((dataset_dir / group / "metadata.json").read_text())
        for group in config["data"]["source_groups"]
    ]


def disease_predictions(
    disease: str,
    x: dict[str, np.ndarray],
    source: dict[str, np.ndarray],
) -> list[bool | None]:
    if disease == "diabetes":
        pred = (x["HE_glu"] >= 126) | (x["HE_HbA1c"] >= 6.5)
        missing = np.zeros_like(pred, dtype=bool)
    elif disease == "hypertension":
        pred = (x["HE_sbp"] >= 140) | (x["HE_dbp"] >= 90)
        missing = np.zeros_like(pred, dtype=bool)
    elif disease == "dyslipidemia":
        sex = require_source(source, "sex", disease)
        pred = (
            (x["HE_chol"] >= 240)
            | (x["HE_TG"] >= 200)
            | ((sex == 1) & (x["HE_HDL_st2"] < 40))
            | ((sex == 2) & (x["HE_HDL_st2"] < 50))
        )
        missing = np.isnan(sex)
    elif disease == "liver_disease":
        pred = (x["HE_ast"] >= 40) | (x["HE_alt"] >= 40)
        missing = np.zeros_like(pred, dtype=bool)
    elif disease == "hepatitis_b":
        pred = x["HE_hepaB"] == 2
        missing = np.zeros_like(pred, dtype=bool)
    elif disease == "hepatitis_c":
        pred = x["HE_hepaC"] == 2
        missing = np.zeros_like(pred, dtype=bool)
    elif disease == "kidney_disease":
        age = require_source(source, "age", disease)
        sex = require_source(source, "sex", disease)
        crea = x["HE_crea"]
        female_egfr = (
            141
            * np.minimum(crea / 0.7, 1) ** (-0.329)
            * np.maximum(crea / 0.7, 1) ** (-1.209)
            * 0.993**age
            * 1.018
        )
        male_egfr = (
            141
            * np.minimum(crea / 0.9, 1) ** (-0.411)
            * np.maximum(crea / 0.9, 1) ** (-1.209)
            * 0.993**age
        )
        egfr = np.full_like(crea, np.nan, dtype=np.float64)
        egfr[sex == 2] = female_egfr[sex == 2]
        egfr[sex == 1] = male_egfr[sex == 1]
        pred = (age > 18) & (egfr < 60)
        missing = np.isnan(age) | np.isnan(sex)
    elif disease == "anemia":
        sex = require_source(source, "sex", disease)
        pred = ((sex == 1) & (x["HE_HB"] < 13)) | ((sex == 2) & (x["HE_HB"] < 12))
        missing = np.isnan(sex)
    else:
        raise ValueError(f"Unsupported disease: {disease}")
    return [None if is_missing else bool(value) for value, is_missing in zip(pred, missing)]


def require_source(
    source: dict[str, np.ndarray],
    name: str,
    disease: str,
) -> np.ndarray:
    if name not in source:
        raise ValueError(f"{disease} evaluation requires source feature {name!r}.")
    return source[name]


def load_test_labels(config: dict[str, Any], disease: str) -> list[bool | None]:
    dataset_name = str(config["data"]["dataset_name"])
    suffix = dataset_name.removeprefix("harmonized_")
    path = Path("datasets/harmonized") / f"disease_labels_harmonized_{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    split_path = (
        Path(config["data"]["dataset_root"])
        / dataset_name
        / disease
        / "split.csv"
    )
    test_indices = []
    with split_path.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["split"] == "test":
                test_indices.append(int(row["row_index"]))

    column = f"disease_{disease}"
    labels_by_index: dict[int, bool | None] = {}
    with path.open(newline="", errors="replace") as file:
        reader = csv.DictReader(file)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"{path} does not contain {column}.")
        for row in reader:
            idx = int(row["row_index"])
            labels_by_index[idx] = parse_label(row[column])
    return [labels_by_index[idx] for idx in test_indices]


def parse_label(value: str) -> bool | None:
    value = value.strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def binary_metrics(predictions: list[bool | None], labels: list[bool | None]) -> dict[str, Any]:
    tp = tn = fp = fn = missing_label = missing_prediction = 0
    for pred, label in zip(predictions, labels):
        if label is None:
            missing_label += 1
        elif pred is None:
            missing_prediction += 1
        elif pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
        elif not pred and label:
            fn += 1
        else:
            tn += 1
    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "missing_label": missing_label,
        "missing_prediction": missing_prediction,
        "n_evaluable": tp + tn + fp + fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def binary_auroc(scores: list[float | None], labels: list[bool | None]) -> float | None:
    observed = [
        (float(score), bool(label))
        for score, label in zip(scores, labels)
        if score is not None and label is not None
    ]
    n_pos = sum(1 for _, label in observed if label)
    n_neg = sum(1 for _, label in observed if not label)
    if n_pos == 0 or n_neg == 0:
        return None

    sorted_observed = sorted(observed, key=lambda item: item[0])
    rank_sum_pos = 0.0
    rank = 1
    idx = 0
    while idx < len(sorted_observed):
        end = idx + 1
        while end < len(sorted_observed) and sorted_observed[end][0] == sorted_observed[idx][0]:
            end += 1
        average_rank = 0.5 * (rank + end)
        positives_in_tie = sum(1 for _, label in sorted_observed[idx:end] if label)
        rank_sum_pos += positives_in_tie * average_rank
        rank += end - idx
        idx = end

    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def safe_divide(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
