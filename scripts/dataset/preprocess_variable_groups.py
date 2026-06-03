"""Preprocess harmonized KNHANES variable groups for conditional VAE training.

The script intentionally uses only the Python standard library so it can run in
this workspace before the training environment is fully set up.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import struct
import sys
from array import array
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any
from zipfile import ZipFile
import xml.etree.ElementTree as ET

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - JSON configs still work.
    yaml = None


NUMERICAL_TYPES = {"numerical", "generated_numerical"}
CATEGORICAL_TYPES = {"categorical"}
SKIP_TYPES = {"identifier", "survey_design", "string", ""}
MISSING_STRINGS = {"", "na", "nan", "none", "null", "."}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    variable_type: str
    group: str


@dataclass
class QuantileState:
    feature: str
    n_valid_train: int
    constant: bool
    quantiles: list[float]
    references: list[float]
    normal_values: list[float]


@dataclass(frozen=True)
class HarmonizedRowFilter:
    surveys: tuple[str, ...] | None = None
    year_min: float | None = None
    year_max: float | None = None


@dataclass(frozen=True)
class HarmonizedDatasetSpec:
    path: Path
    name: str
    row_filter: HarmonizedRowFilter | None = None


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.harmonized_dataset is not None:
        config["harmonized_dataset"] = args.harmonized_dataset
    if args.disease_label is not None:
        config["disease_label"] = args.disease_label
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.source_groups is not None:
        config["source_groups"] = args.source_groups
    if args.target_group is not None:
        config["target_group"] = args.target_group
    if args.variable_groups is not None:
        config["variable_groups"] = args.variable_groups
    if (
        args.dataset_name is not None
        or args.survey is not None
        or args.year_min is not None
        or args.year_max is not None
    ):
        config.pop("harmonized_views", None)
    if args.dataset_name is not None:
        config["dataset_name"] = args.dataset_name
    if args.survey is not None:
        config["survey"] = args.survey
    if args.year_min is not None:
        config["year_min"] = args.year_min
    if args.year_max is not None:
        config["year_max"] = args.year_max

    run_preprocessing(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess source/target variable groups for CVAE training."
    )
    parser.add_argument("--config", required=True, help="YAML or JSON config path.")
    parser.add_argument("--harmonized-dataset", help="Override harmonized CSV path.")
    parser.add_argument("--dataset-name", help="Override output dataset/view name.")
    parser.add_argument(
        "--survey",
        nargs="+",
        help="Keep only these survey values, e.g. knhanes or nhanes.",
    )
    parser.add_argument("--year-min", type=float, help="Keep rows with year >= this value.")
    parser.add_argument("--year-max", type=float, help="Keep rows with year <= this value.")
    parser.add_argument("--disease-label", help="Override disease label CSV path.")
    parser.add_argument("--output-dir", help="Override output root directory.")
    parser.add_argument(
        "--source-groups",
        nargs="+",
        help="Override source variable group files, e.g. questionnaire_without_disease.txt dietary.txt.",
    )
    parser.add_argument("--target-group", help="Override target variable group file.")
    parser.add_argument(
        "--variable-groups",
        nargs="+",
        help="Override group-wise variable group files.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to read YAML configs.")
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping.")
    return data


def run_preprocessing(config: dict[str, Any]) -> None:
    mode = config.get("mode")
    if mode is None:
        mode = "variable_groups" if "variable_groups" in config else "source_target"
    if mode == "variable_groups":
        run_variable_group_preprocessing(config)
    elif mode == "source_target":
        run_source_target_preprocessing(config)
    else:
        raise ValueError(f"Unsupported preprocessing mode: {mode!r}")


def run_variable_group_preprocessing(config: dict[str, Any]) -> None:
    docs_dir = Path(config.get("docs_dir", "datasets/docs"))
    output_root = Path(config.get("output_dir", "datasets/preprocessed"))
    seed = int(config.get("seed", 42))

    harmonized_specs = resolve_harmonized_specs(config)
    variable_groups = list(required(config, "variable_groups"))

    quantile_config = dict(config.get("gaussian_quantile", {}))
    n_quantiles = int(quantile_config.get("n_quantiles", 1000))
    eps = float(quantile_config.get("clip_epsilon", 1e-6))
    sample_weight_enabled = bool(
        config.get("sample_weight", {}).get("enabled", False)
    )
    if n_quantiles < 2:
        raise ValueError("gaussian_quantile.n_quantiles must be >= 2.")
    if not (0.0 < eps < 0.5):
        raise ValueError("gaussian_quantile.clip_epsilon must be in (0, 0.5).")

    variable_records = read_xlsx_records(docs_dir / "knhanes_variable_dictionary.xlsx")
    value_records = read_xlsx_records(docs_dir / "knhanes_value_dictionary.xlsx")
    disease_derivations = read_disease_derivations(docs_dir / "disease_derivation.xlsx")
    variable_meta = build_variable_meta(variable_records)
    value_meta = build_value_meta(value_records)

    for harmonized_spec in harmonized_specs:
        disease_label_path = default_disease_label_path(harmonized_spec)
        for variable_group in variable_groups:
            specs, warnings = resolve_feature_specs(
                docs_dir=docs_dir,
                group_names=[variable_group],
                variable_meta=variable_meta,
                allow_skipped=True,
            )
            cat_specs, num_specs = split_feature_specs(specs)
            selected_features = expand_generated_dependencies(
                unique_names([*cat_specs, *num_specs]),
                value_meta,
            )
            rows = read_selected_harmonized_rows(
                harmonized_spec.path,
                selected_features,
                row_filter=harmonized_spec.row_filter,
            )
            split = make_split(rows["__split_year__"], seed=seed)

            cat_values = encode_categorical_matrix(rows, cat_specs, value_meta)
            num_raw, num_mask = encode_numerical_matrix(rows, num_specs, value_meta)
            num_values, quantiles = gaussian_quantile_matrix(
                num_raw,
                num_mask,
                split,
                feature_names=[x.name for x in num_specs],
                n_quantiles=n_quantiles,
                eps=eps,
            )

            group_name = group_stem(variable_group)
            output_dir = output_root / harmonized_spec.name / group_name
            output_dir.mkdir(parents=True, exist_ok=True)
            write_npy(output_dir / "cat.npy", cat_values, dtype="int64")
            write_npy(output_dir / "num.npy", num_values, dtype="float32")
            write_npy(output_dir / "num_valid_mask.npy", num_mask, dtype="float32")
            write_split_csv(output_dir / "split.csv", split, rows["__split_year__"])
            write_row_metadata_csv(output_dir / "row_metadata.csv", rows)

            sample_weight_metadata = None
            if sample_weight_enabled and disease_label_path.exists():
                disease_label_column = f"disease_{group_name}"
                if disease_label_column in read_csv_header(disease_label_path):
                    labels = read_disease_labels(disease_label_path, disease_label_column)
                    if len(labels) != len(split):
                        raise ValueError(
                            f"Row count mismatch: {harmonized_spec.name} has {len(split)} rows, "
                            f"{disease_label_path} has {len(labels)} labels."
                        )
                    sample_weight, label_counts, positive_weight = make_sample_weights(labels)
                    write_npy_vector(
                        output_dir / "sample_weight.npy",
                        sample_weight,
                        dtype="float32",
                    )
                    sample_weight_metadata = {
                        "disease_label": str(disease_label_path),
                        "disease_label_column": disease_label_column,
                        "policy": "positive=n_false/n_true, false=1, missing=1",
                        "positive_weight": positive_weight,
                        "label_counts": label_counts,
                    }

            if group_name in disease_derivations:
                derivation_metadata, derivation_warnings = validate_disease_derivation(
                    target_disease=group_name,
                    target_group_variables=[spec.name for spec in specs],
                    disease_derivations=disease_derivations,
                )
            else:
                derivation_metadata, derivation_warnings = {}, []
            metadata = {
                "mode": "variable_groups",
                "dataset_name": harmonized_spec.name,
                "harmonized_dataset": str(harmonized_spec.path),
                "harmonized_filter": harmonized_filter_to_json(
                    harmonized_spec.row_filter
                ),
                "variable_group": variable_group,
                "group_name": group_name,
                "n_rows": len(split),
                "features": {
                    "cat": [x.name for x in cat_specs],
                    "num": [x.name for x in num_specs],
                },
                "category_sizes": category_sizes_for(cat_specs, value_meta),
                "split": dict(Counter(split)),
                "split_policy": {
                    "seed": seed,
                    "valid_test_source": "KNHANES rows with year == 2024",
                    "valid_fraction_of_2024": 0.25,
                    "test_fraction_of_2024": 0.25,
                },
                "gaussian_quantile": {
                    "n_quantiles": n_quantiles,
                    "clip_epsilon": eps,
                    "fit_split": "train",
                    "features": [quantile_to_json(x) for x in quantiles],
                },
                "disease_derivation": derivation_metadata,
                "sample_weight": sample_weight_metadata,
                "warnings": [*warnings, *derivation_warnings],
            }
            (output_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
            )
            print(
                f"Wrote {output_dir} "
                f"cat=({len(cat_values)}, {len(cat_specs)}) "
                f"num=({len(num_values)}, {len(num_specs)})",
                flush=True,
            )


def run_source_target_preprocessing(config: dict[str, Any]) -> None:
    docs_dir = Path(config.get("docs_dir", "datasets/docs"))
    harmonized_specs = resolve_harmonized_specs(config)
    output_root = Path(config.get("output_dir", "datasets/preprocessed"))
    source_groups = list(required(config, "source_groups"))
    target_group = str(required(config, "target_group"))
    target_disease = group_stem(target_group)
    disease_label_column = config.get("disease_label_column") or f"disease_{target_disease}"
    seed = int(config.get("seed", 42))

    quantile_config = dict(config.get("gaussian_quantile", {}))
    n_quantiles = int(quantile_config.get("n_quantiles", 1000))
    eps = float(quantile_config.get("clip_epsilon", 1e-6))
    sample_weight_enabled = bool(
        config.get("sample_weight", {}).get("enabled", False)
    )
    if n_quantiles < 2:
        raise ValueError("gaussian_quantile.n_quantiles must be >= 2.")
    if not (0.0 < eps < 0.5):
        raise ValueError("gaussian_quantile.clip_epsilon must be in (0, 0.5).")

    variable_records = read_xlsx_records(docs_dir / "knhanes_variable_dictionary.xlsx")
    value_records = read_xlsx_records(docs_dir / "knhanes_value_dictionary.xlsx")
    disease_derivations = read_disease_derivations(docs_dir / "disease_derivation.xlsx")
    variable_meta = build_variable_meta(variable_records)
    value_meta = build_value_meta(value_records)

    source_specs, source_warnings = resolve_feature_specs(
        docs_dir=docs_dir,
        group_names=source_groups,
        variable_meta=variable_meta,
        allow_skipped=True,
    )
    target_specs, target_warnings = resolve_feature_specs(
        docs_dir=docs_dir,
        group_names=[target_group],
        variable_meta=variable_meta,
        allow_skipped=False,
    )
    source_cat, source_num = split_feature_specs(source_specs)
    target_cat, target_num = split_feature_specs(target_specs)
    target_group_variables = [spec.name for spec in target_specs]
    derivation_metadata, derivation_warnings = validate_disease_derivation(
        target_disease=target_disease,
        target_group_variables=target_group_variables,
        disease_derivations=disease_derivations,
    )

    category_sizes = {
        "source": category_sizes_for(source_cat, value_meta),
        "target": category_sizes_for(target_cat, value_meta),
    }

    selected_features = expand_generated_dependencies(
        unique_names([*source_cat, *source_num, *target_cat, *target_num]),
        value_meta,
    )

    for harmonized_spec in harmonized_specs:
        disease_label_path = Path(
            config.get("disease_label") or default_disease_label_path(harmonized_spec)
        )
        rows = read_selected_harmonized_rows(
            harmonized_spec.path,
            selected_features,
            row_filter=harmonized_spec.row_filter,
        )
        labels = read_disease_labels(disease_label_path, disease_label_column)
        n_rows = len(rows["__split_year__"])
        if n_rows != len(labels):
            raise ValueError(
                f"Row count mismatch: {harmonized_spec.name} has {n_rows} rows, "
                f"{disease_label_path} has {len(labels)} labels."
            )

        split = make_split(rows["__split_year__"], seed=seed)

        source_cat_values = encode_categorical_matrix(rows, source_cat, value_meta)
        target_cat_values = encode_categorical_matrix(rows, target_cat, value_meta)
        source_num_raw, source_num_mask = encode_numerical_matrix(
            rows, source_num, value_meta
        )
        target_num_raw, target_num_mask = encode_numerical_matrix(
            rows, target_num, value_meta
        )

        source_num_values, source_quantiles = gaussian_quantile_matrix(
            source_num_raw,
            source_num_mask,
            split,
            feature_names=[x.name for x in source_num],
            n_quantiles=n_quantiles,
            eps=eps,
        )
        target_num_values, target_quantiles = gaussian_quantile_matrix(
            target_num_raw,
            target_num_mask,
            split,
            feature_names=[x.name for x in target_num],
            n_quantiles=n_quantiles,
            eps=eps,
        )

        sample_weight = None
        label_counts = None
        positive_weight = None
        if sample_weight_enabled:
            sample_weight, label_counts, positive_weight = make_sample_weights(labels)

        output_dir = (
            output_root
            / harmonized_spec.name
            / f"target_{target_disease}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        write_npy(output_dir / "source_cat.npy", source_cat_values, dtype="int64")
        write_npy(output_dir / "source_num.npy", source_num_values, dtype="float32")
        write_npy(output_dir / "source_num_valid_mask.npy", source_num_mask, dtype="float32")
        write_npy(output_dir / "target_cat.npy", target_cat_values, dtype="int64")
        write_npy(output_dir / "target_num.npy", target_num_values, dtype="float32")
        write_npy(output_dir / "target_num_valid_mask.npy", target_num_mask, dtype="float32")
        if sample_weight is not None:
            write_npy_vector(output_dir / "sample_weight.npy", sample_weight, dtype="float32")
        write_split_csv(output_dir / "split.csv", split, rows["__split_year__"])
        write_row_metadata_csv(output_dir / "row_metadata.csv", rows)

        metadata = {
            "dataset_name": harmonized_spec.name,
            "harmonized_dataset": str(harmonized_spec.path),
            "harmonized_filter": harmonized_filter_to_json(
                harmonized_spec.row_filter
            ),
            "disease_label": str(disease_label_path),
            "target_disease": target_disease,
            "disease_label_column": disease_label_column,
            "source_groups": source_groups,
            "target_group": target_group,
            "disease_derivation": derivation_metadata,
            "n_rows": len(split),
            "features": {
                "source_cat": [x.name for x in source_cat],
                "source_num": [x.name for x in source_num],
                "target_cat": [x.name for x in target_cat],
                "target_num": [x.name for x in target_num],
            },
            "category_sizes": category_sizes,
            "split": dict(Counter(split)),
            "split_policy": {
                "seed": seed,
                "valid_test_source": "KNHANES rows with year == 2024",
                "valid_fraction_of_2024": 0.25,
                "test_fraction_of_2024": 0.25,
            },
            "gaussian_quantile": {
                "n_quantiles": n_quantiles,
                "clip_epsilon": eps,
                "fit_split": "train",
                "source": [quantile_to_json(x) for x in source_quantiles],
                "target": [quantile_to_json(x) for x in target_quantiles],
            },
            "sample_weight": (
                {
                    "policy": "positive=n_false/n_true, false=1, missing=1",
                    "positive_weight": positive_weight,
                    "label_counts": label_counts,
                }
                if sample_weight_enabled
                else None
            ),
            "warnings": {
                "source": source_warnings,
                "target": [*target_warnings, *derivation_warnings],
            },
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
        )

        print(f"Wrote {output_dir}")
        print(f"Rows: {len(split)}; split: {dict(Counter(split))}")
        if positive_weight is not None:
            print(
                f"Sample positive weight for {disease_label_column}: "
                f"{positive_weight:.6g}"
            )


def required(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise KeyError(f"Missing required config key: {key}")
    return config[key]


def resolve_harmonized_paths(config: dict[str, Any]) -> list[Path]:
    return [spec.path for spec in resolve_harmonized_specs(config)]


def resolve_harmonized_specs(config: dict[str, Any]) -> list[HarmonizedDatasetSpec]:
    if "harmonized_views" in config:
        specs = resolve_harmonized_view_specs(config)
    elif "harmonized_datasets" in config:
        specs = [
            HarmonizedDatasetSpec(path=Path(path), name=Path(path).stem)
            for path in config["harmonized_datasets"]
        ]
    elif "harmonized_dataset" in config:
        path = Path(config["harmonized_dataset"])
        row_filter = build_harmonized_row_filter(config)
        specs = [
            HarmonizedDatasetSpec(
                path=path,
                name=str(config.get("dataset_name") or infer_harmonized_name(path, row_filter)),
                row_filter=row_filter,
            )
        ]
    else:
        specs = [
            HarmonizedDatasetSpec(path=path, name=path.stem)
            for path in sorted(
                Path(config.get("harmonized_dir", "datasets/harmonized")).glob(
                    "harmonized_*.csv"
                )
            )
        ]
    if not specs:
        raise ValueError("No harmonized datasets were found.")
    missing = [str(spec.path) for spec in specs if not spec.path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing harmonized datasets: {missing}")
    duplicate_names = [
        name for name, count in Counter(spec.name for spec in specs).items() if count > 1
    ]
    if duplicate_names:
        raise ValueError(f"Duplicate harmonized dataset names: {duplicate_names}")
    return specs


def resolve_harmonized_view_specs(config: dict[str, Any]) -> list[HarmonizedDatasetSpec]:
    views = config["harmonized_views"]
    if not isinstance(views, list):
        raise ValueError("harmonized_views must be a list.")
    base_path = config.get("harmonized_dataset")
    specs: list[HarmonizedDatasetSpec] = []
    for idx, view in enumerate(views):
        if not isinstance(view, dict):
            raise ValueError(f"harmonized_views[{idx}] must be a mapping.")
        path_value = view.get("harmonized_dataset") or view.get("path") or base_path
        if path_value is None:
            raise KeyError(
                "harmonized_views entries require path/harmonized_dataset "
                "when top-level harmonized_dataset is absent."
            )
        path = Path(path_value)
        row_filter = build_harmonized_row_filter(view)
        name = str(view.get("name") or view.get("dataset_name") or infer_harmonized_name(path, row_filter))
        specs.append(HarmonizedDatasetSpec(path=path, name=name, row_filter=row_filter))
    return specs


def build_harmonized_row_filter(config: dict[str, Any]) -> HarmonizedRowFilter | None:
    surveys_value = first_config_value(config, "surveys", "survey")
    surveys = parse_survey_filter(surveys_value)
    year_min = parse_optional_float(first_config_value(config, "year_min", "min_year", "start_year"))
    year_max = parse_optional_float(first_config_value(config, "year_max", "max_year", "end_year"))
    if surveys is None and year_min is None and year_max is None:
        return None
    return HarmonizedRowFilter(surveys=surveys, year_min=year_min, year_max=year_max)


def first_config_value(config: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in config:
            return config[key]
    return None


def parse_survey_filter(value: Any) -> tuple[str, ...] | None:
    if value is None or value == "":
        return None
    values = value if isinstance(value, list) else [value]
    normalized = tuple(
        normalize_survey_name(item)
        for item in values
        if normalize_survey_name(item)
    )
    return normalized or None


def infer_harmonized_name(path: Path, row_filter: HarmonizedRowFilter | None) -> str:
    if (
        row_filter is not None
        and row_filter.surveys is not None
        and len(row_filter.surveys) == 1
        and row_filter.year_min is not None
        and row_filter.year_max is not None
    ):
        return (
            f"harmonized_{row_filter.surveys[0]}_"
            f"{format_year_for_name(row_filter.year_min)}_"
            f"{format_year_for_name(row_filter.year_max)}"
        )
    return path.stem


def format_year_for_name(year: float) -> str:
    return str(int(year)) if float(year).is_integer() else str(year).replace(".", "p")


def default_disease_label_path(spec: HarmonizedDatasetSpec) -> Path:
    return spec.path.with_name(f"disease_labels_{spec.name}.csv")


def harmonized_filter_to_json(
    row_filter: HarmonizedRowFilter | None,
) -> dict[str, Any] | None:
    if row_filter is None:
        return None
    return {
        "surveys": list(row_filter.surveys) if row_filter.surveys is not None else None,
        "year_min": row_filter.year_min,
        "year_max": row_filter.year_max,
    }


def row_matches_harmonized_filter(
    row: dict[str, Any],
    row_filter: HarmonizedRowFilter | None,
) -> bool:
    if row_filter is None:
        return True
    if row_filter.surveys is not None:
        survey = normalize_survey_name(row.get("survey", ""))
        if not survey:
            survey = normalize_survey_name(row.get("source", ""))
        if survey not in row_filter.surveys:
            return False
    if row_filter.year_min is not None or row_filter.year_max is not None:
        year = parse_optional_float(row.get("year", ""))
        if year is None:
            return False
        if row_filter.year_min is not None and year < row_filter.year_min:
            return False
        if row_filter.year_max is not None and year > row_filter.year_max:
            return False
    return True


def normalize_survey_name(value: Any) -> str:
    key = normalize_value_key(value).lower()
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


def read_csv_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        return next(reader, [])


def group_stem(group_name: str) -> str:
    return Path(group_name).stem


def resolve_group_file(docs_dir: Path, group_name: str) -> Path:
    candidate = docs_dir / group_name
    if candidate.exists():
        return candidate
    if not group_name.endswith(".txt"):
        txt_candidate = docs_dir / f"{group_name}.txt"
        if txt_candidate.exists():
            return txt_candidate
    raise FileNotFoundError(f"Variable group file not found for {group_name!r}.")


def read_group_variables(docs_dir: Path, group_name: str) -> list[str]:
    path = resolve_group_file(docs_dir, group_name)
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def resolve_feature_specs(
    *,
    docs_dir: Path,
    group_names: list[str],
    variable_meta: dict[str, dict[str, str]],
    allow_skipped: bool,
) -> tuple[list[FeatureSpec], list[str]]:
    specs: list[FeatureSpec] = []
    warnings: list[str] = []
    seen: set[str] = set()

    for group_name in group_names:
        for variable in read_group_variables(docs_dir, group_name):
            if variable in seen:
                warnings.append(f"Duplicate variable skipped: {variable}")
                continue
            seen.add(variable)
            meta = variable_meta.get(variable)
            if meta is None:
                message = f"Variable not found in variable dictionary: {variable}"
                if allow_skipped:
                    warnings.append(message)
                    continue
                raise ValueError(message)

            variable_type = meta.get("variable_type", "")
            excluded = meta.get("exclude", "").lower() in {"true", "exclude", "yes", "1"}
            if excluded or variable_type in SKIP_TYPES:
                message = f"Skipped {variable} with type={variable_type!r}, exclude={excluded}"
                if allow_skipped:
                    warnings.append(message)
                    continue
                raise ValueError(message)
            if variable_type not in NUMERICAL_TYPES | CATEGORICAL_TYPES:
                message = f"Unsupported variable type for {variable}: {variable_type!r}"
                if allow_skipped:
                    warnings.append(message)
                    continue
                raise ValueError(message)
            specs.append(FeatureSpec(variable, variable_type, group_stem(group_name)))

    return specs, warnings


def split_feature_specs(
    specs: list[FeatureSpec],
) -> tuple[list[FeatureSpec], list[FeatureSpec]]:
    cat = [x for x in specs if x.variable_type in CATEGORICAL_TYPES]
    num = [x for x in specs if x.variable_type in NUMERICAL_TYPES]
    return cat, num


def unique_names(specs: list[FeatureSpec]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        if spec.name not in seen:
            seen.add(spec.name)
            names.append(spec.name)
    return names


def expand_generated_dependencies(
    feature_names: list[str],
    value_meta: dict[str, dict[str, Any]],
) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        if name not in seen:
            seen.add(name)
            names.append(name)

    for feature in feature_names:
        rule = value_meta.get(feature, {}).get("generated_rule")
        if rule:
            add(rule["source1"])
            add(rule["source2"])
        else:
            add(feature)
    return names


def read_selected_harmonized_rows(
    path: Path,
    selected_features: list[str],
    *,
    row_filter: HarmonizedRowFilter | None = None,
) -> dict[str, list[str]]:
    data: dict[str, list[str]] = {
        "__split_year__": [],
        "__id__": [],
        "__id_fam__": [],
        "__survey__": [],
        "__source__": [],
        "__sex__": [],
        "__age__": [],
    }
    for feature in selected_features:
        data[feature] = []

    with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = [x for x in selected_features if x not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing selected columns: {missing[:20]}")
        if "year" not in reader.fieldnames:
            raise ValueError(f"{path} must include a year column for splitting.")
        if row_filter is not None and row_filter.surveys is not None and "survey" not in reader.fieldnames:
            raise ValueError(f"{path} must include a survey column for survey filtering.")
        for row in reader:
            if not row_matches_harmonized_filter(row, row_filter):
                continue
            data["__split_year__"].append(row.get("year", ""))
            data["__id__"].append(row.get("ID", ""))
            data["__id_fam__"].append(row.get("ID_fam", ""))
            data["__survey__"].append(row.get("survey", ""))
            data["__source__"].append(row.get("source", ""))
            data["__sex__"].append(row.get("sex", ""))
            data["__age__"].append(row.get("age", ""))
            for feature in selected_features:
                data[feature].append(row.get(feature, ""))
    if not data["__split_year__"]:
        raise ValueError(
            f"{path} yielded no rows after applying filter "
            f"{harmonized_filter_to_json(row_filter)}."
        )
    return data


def read_disease_labels(path: Path, column: str) -> list[str | None]:
    labels: list[str | None] = []
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"{path} does not contain disease label column {column!r}.")
        for idx, row in enumerate(reader):
            if "row_index" in row and row["row_index"].strip():
                try:
                    if int(float(row["row_index"])) != idx:
                        raise ValueError(
                            f"{path} row_index mismatch at row {idx}: {row['row_index']}"
                        )
                except ValueError:
                    raise
            labels.append(parse_label(row.get(column, "")))
    return labels


def read_disease_derivations(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    records = read_xlsx_records(path)
    derivations: dict[str, dict[str, str]] = {}
    for record in records:
        disease_name = record.get("disease_name", "").strip()
        if not disease_name:
            continue
        derivations[disease_name] = {
            "target_indicators": record.get("target_indicators", "").strip(),
            "conditions": record.get("conditions", "").strip(),
        }
    return derivations


def validate_disease_derivation(
    *,
    target_disease: str,
    target_group_variables: list[str],
    disease_derivations: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    derivation = disease_derivations.get(target_disease)
    if derivation is None:
        warnings.append(f"disease_derivation.xlsx has no row for {target_disease!r}.")
        return {}, warnings

    derivation_indicators = [
        value.strip()
        for value in derivation.get("target_indicators", "").split(",")
        if value.strip()
    ]
    group_set = set(target_group_variables)
    derivation_set = set(derivation_indicators)
    missing_from_group = sorted(derivation_set - group_set)
    extra_in_group = sorted(group_set - derivation_set)
    if missing_from_group:
        warnings.append(
            "Target group does not include derivation indicators: "
            + ", ".join(missing_from_group)
        )
    if extra_in_group:
        warnings.append(
            "Target group includes variables not listed in derivation indicators: "
            + ", ".join(extra_in_group)
        )
    return {
        "target_indicators": derivation_indicators,
        "conditions": derivation.get("conditions", ""),
    }, warnings


def parse_label(value: str | None) -> str | None:
    text = "" if value is None else str(value).strip().lower()
    if text == "true":
        return "true"
    if text == "false":
        return "false"
    return None


def make_split(year_values: list[str], *, seed: int) -> list[str]:
    split = ["train"] * len(year_values)
    indices_2024 = [
        i for i, value in enumerate(year_values) if parse_optional_float(value) == 2024.0
    ]
    rng = random.Random(seed)
    rng.shuffle(indices_2024)
    n_valid = int(len(indices_2024) * 0.25)
    n_test = int(len(indices_2024) * 0.25)
    for idx in indices_2024[:n_valid]:
        split[idx] = "valid"
    for idx in indices_2024[n_valid : n_valid + n_test]:
        split[idx] = "test"
    return split


def make_sample_weights(labels: list[str | None]) -> tuple[list[float], dict[str, int], float]:
    counts = {
        "true": sum(1 for x in labels if x == "true"),
        "false": sum(1 for x in labels if x == "false"),
        "missing": sum(1 for x in labels if x is None),
    }
    if counts["true"] == 0:
        positive_weight = 1.0
    else:
        positive_weight = counts["false"] / counts["true"]
    weights = [
        positive_weight if label == "true" else 1.0
        for label in labels
    ]
    return weights, counts, positive_weight


def encode_categorical_matrix(
    rows: dict[str, list[str]],
    specs: list[FeatureSpec],
    value_meta: dict[str, dict[str, Any]],
) -> list[list[int]]:
    if not specs:
        return [[] for _ in rows["__split_year__"]]
    encoded_columns: list[list[int]] = []
    for spec in specs:
        mapping = value_meta.get(spec.name, {}).get("categorical_mapping", {})
        column: list[int] = []
        for raw in rows[spec.name]:
            key = normalize_value_key(raw)
            column.append(int(mapping.get(key, 0)))
        encoded_columns.append(column)
    return transpose_columns(encoded_columns)


def encode_numerical_matrix(
    rows: dict[str, list[str]],
    specs: list[FeatureSpec],
    value_meta: dict[str, dict[str, Any]],
) -> tuple[list[list[float]], list[list[float]]]:
    if not specs:
        empty = [[] for _ in rows["__split_year__"]]
        return empty, empty

    value_columns: list[list[float]] = []
    mask_columns: list[list[float]] = []
    for spec in specs:
        missing_values = value_meta.get(spec.name, {}).get("numerical_missing_values", set())
        generated_rule = value_meta.get(spec.name, {}).get("generated_rule")
        values: list[float] = []
        masks: list[float] = []
        for row_idx in range(len(rows["__split_year__"])):
            if generated_rule:
                value = compute_generated_value(
                    rows=rows,
                    row_idx=row_idx,
                    rule=generated_rule,
                    value_meta=value_meta,
                )
                key = ""
            else:
                raw = rows[spec.name][row_idx]
                key = normalize_value_key(raw)
                value = parse_optional_float(raw)
            if key in missing_values or value is None or not math.isfinite(value):
                values.append(0.0)
                masks.append(0.0)
            else:
                values.append(float(value))
                masks.append(1.0)
        value_columns.append(values)
        mask_columns.append(masks)
    return transpose_columns(value_columns), transpose_columns(mask_columns)


def compute_generated_value(
    *,
    rows: dict[str, list[str]],
    row_idx: int,
    rule: dict[str, Any],
    value_meta: dict[str, dict[str, Any]],
) -> float | None:
    source_values: list[float] = []
    for source_name in ["source1", "source2"]:
        raw = rows[rule[source_name]][row_idx]
        key = normalize_value_key(raw)
        value = parse_optional_float(raw)
        missing_values = value_meta.get(rule[source_name], {}).get(
            "numerical_missing_values", set()
        )
        if key in missing_values or value is None or not math.isfinite(value):
            return None
        source_values.append(float(value))
    return (
        source_values[0] * float(rule["source1_multiplier"])
        + source_values[1] * float(rule["source2_multiplier"])
    )


def gaussian_quantile_matrix(
    values: list[list[float]],
    masks: list[list[float]],
    split: list[str],
    *,
    feature_names: list[str],
    n_quantiles: int,
    eps: float,
) -> tuple[list[list[float]], list[QuantileState]]:
    if not values or not values[0]:
        return values, []

    n_rows = len(values)
    n_cols = len(values[0])
    transformed = [[0.0] * n_cols for _ in range(n_rows)]
    states: list[QuantileState] = []

    for col_idx in range(n_cols):
        train_values = sorted(
            values[row_idx][col_idx]
            for row_idx in range(n_rows)
            if split[row_idx] == "train" and masks[row_idx][col_idx] > 0.0
        )
        state = fit_quantile_state(
            feature=feature_names[col_idx],
            train_values=train_values,
            n_quantiles=n_quantiles,
            eps=eps,
        )
        states.append(state)
        for row_idx in range(n_rows):
            if masks[row_idx][col_idx] > 0.0:
                transformed[row_idx][col_idx] = transform_with_state(
                    values[row_idx][col_idx], state, eps=eps
                )
    return transformed, states


def fit_quantile_state(
    *,
    feature: str,
    train_values: list[float],
    n_quantiles: int,
    eps: float,
) -> QuantileState:
    if not train_values:
        return QuantileState(feature, 0, True, [], [], [])
    if train_values[0] == train_values[-1]:
        return QuantileState(feature, len(train_values), True, [train_values[0]], [0.5], [0.0])

    actual_n_quantiles = min(n_quantiles, len(train_values))
    references = [
        i / (actual_n_quantiles - 1)
        for i in range(actual_n_quantiles)
    ]
    quantiles = [
        interpolated_sorted_value(train_values, reference)
        for reference in references
    ]
    clipped_references = [min(max(reference, eps), 1.0 - eps) for reference in references]
    normal = NormalDist()
    normal_values = [normal.inv_cdf(reference) for reference in clipped_references]
    return QuantileState(
        feature=feature,
        n_valid_train=len(train_values),
        constant=False,
        quantiles=quantiles,
        references=clipped_references,
        normal_values=normal_values,
    )


def interpolated_sorted_value(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def transform_with_state(value: float, state: QuantileState, *, eps: float) -> float:
    if state.constant or not state.quantiles:
        return 0.0
    quantiles = state.quantiles
    if value <= quantiles[0]:
        return state.normal_values[0]
    elif value >= quantiles[-1]:
        return state.normal_values[-1]
    else:
        right = bisect_left(quantiles, value)
        left = max(0, right - 1)
        low_v = quantiles[left]
        high_v = quantiles[right]
        low_normal = state.normal_values[left]
        high_normal = state.normal_values[right]
        if high_v == low_v:
            return (low_normal + high_normal) / 2.0
        return low_normal + ((value - low_v) / (high_v - low_v)) * (
            high_normal - low_normal
        )


def read_xlsx_records(path: Path) -> list[dict[str, str]]:
    rows = read_xlsx_rows(path)
    if not rows:
        raise ValueError(f"XLSX has no rows: {path}")
    header = rows[0]
    records: list[dict[str, str]] = []
    for row in rows[1:]:
        padded = row + [""] * (len(header) - len(row))
        records.append({str(key): str(value) for key, value in zip(header, padded)})
    return records


def read_xlsx_rows(path: Path) -> list[list[str]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rid_key = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"

    with ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                shared_strings.append(
                    "".join(t.text or "" for t in si.findall(".//a:t", ns))
                )

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        sheet = workbook.find("a:sheets/a:sheet", ns)
        if sheet is None:
            raise ValueError(f"XLSX has no sheets: {path}")
        sheet_path = normalize_xlsx_target(rel_targets[sheet.attrib[rid_key]])
        root = ET.fromstring(zf.read(sheet_path))

    rows: list[list[str]] = []
    for row in root.findall("a:sheetData/a:row", ns):
        values: list[str] = []
        for cell in row.findall("a:c", ns):
            col_idx = xlsx_col_idx(cell.attrib.get("r", "A1"))
            while len(values) < col_idx:
                values.append("")
            values.append(xlsx_cell_text(cell, shared_strings, ns))
        if any(value.strip() for value in values):
            rows.append(values)
    return rows


def normalize_xlsx_target(target: str) -> str:
    target = target.lstrip("/")
    return target if target.startswith("xl/") else f"xl/{target}"


def xlsx_col_idx(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha()) or "A"
    value = 0
    for letter in letters:
        value = value * 26 + ord(letter.upper()) - 64
    return value - 1


def xlsx_cell_text(cell: ET.Element, shared_strings: list[str], ns: dict[str, str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(t.text or "" for t in cell.findall(".//a:t", ns))
    value_node = cell.find("a:v", ns)
    value = "" if value_node is None else value_node.text or ""
    if cell_type == "s" and value:
        return shared_strings[int(value)]
    return value


def build_variable_meta(records: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {}
    for record in records:
        name = record.get("variable_name", "").strip()
        if not name:
            continue
        meta[name] = {
            "variable_type": record.get("variable_type", "").strip(),
            "exclude": record.get("exclude", "").strip(),
        }
    return meta


def build_value_meta(records: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"categorical_mapping": {}, "numerical_missing_values": set()}
    )
    for record in records:
        name = record.get("variable_name", "").strip()
        variable_type = record.get("variable_type", "").strip()
        raw_value = normalize_value_key(record.get("value", ""))
        preprocessing = record.get("preprocessing", "").strip()
        preprocessing_lower = preprocessing.lower()
        if not name:
            continue
        if variable_type in CATEGORICAL_TYPES:
            encoded = parse_optional_int(preprocessing)
            if encoded is None:
                encoded = 0
            meta[name]["categorical_mapping"][raw_value] = encoded
        elif variable_type in NUMERICAL_TYPES and preprocessing_lower == "missing":
            meta[name]["numerical_missing_values"].add(raw_value)
        elif variable_type == "generated_numerical" and preprocessing_lower == "valid":
            source1 = record.get("source1", "").strip()
            source2 = record.get("source2", "").strip()
            if source1 and source2:
                meta[name]["generated_rule"] = {
                    "source1": source1,
                    "source1_multiplier": parse_required_float(
                        record.get("source1_multiplier", "1")
                    ),
                    "source2": source2,
                    "source2_multiplier": parse_required_float(
                        record.get("source2_multiplier", "1")
                    ),
                }
    return dict(meta)


def category_sizes_for(
    specs: list[FeatureSpec],
    value_meta: dict[str, dict[str, Any]],
) -> list[int]:
    sizes: list[int] = []
    for spec in specs:
        mapping = value_meta.get(spec.name, {}).get("categorical_mapping", {})
        positive_values = [int(value) for value in mapping.values() if int(value) > 0]
        sizes.append(max(positive_values, default=0) + 1)
    return sizes


def normalize_value_key(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if text.lower() in MISSING_STRINGS:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return format(number, ".15g")


def parse_optional_float(value: Any) -> float | None:
    text = "" if value is None else str(value).strip()
    if text.lower() in MISSING_STRINGS:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_optional_int(value: Any) -> int | None:
    number = parse_optional_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def parse_required_float(value: Any) -> float:
    parsed = parse_optional_float(value)
    if parsed is None:
        raise ValueError(f"Expected a numeric value, got {value!r}.")
    return parsed


def transpose_columns(columns: list[list[Any]]) -> list[list[Any]]:
    if not columns:
        return []
    return [list(row) for row in zip(*columns)]


def write_npy(path: Path, rows: list[list[Any]], *, dtype: str) -> None:
    n_rows = len(rows)
    n_cols = len(rows[0]) if rows else 0
    for row in rows:
        if len(row) != n_cols:
            raise ValueError(f"Ragged matrix cannot be written to {path}.")

    if dtype == "float32":
        descr = "<f4"
        typecode = "f"
        cast = float
    elif dtype == "int64":
        descr = "<i8"
        typecode = "q"
        cast = int
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    header = {
        "descr": descr,
        "fortran_order": False,
        "shape": (n_rows, n_cols),
    }
    header_text = (
        "{'descr': '%s', 'fortran_order': False, 'shape': (%d, %d), }"
        % (header["descr"], n_rows, n_cols)
    )
    header_bytes = header_text.encode("latin1")
    padding = 16 - ((10 + len(header_bytes) + 1) % 16)
    header_bytes += b" " * padding + b"\n"

    with path.open("wb") as f:
        f.write(b"\x93NUMPY")
        f.write(bytes([1, 0]))
        f.write(struct.pack("<H", len(header_bytes)))
        f.write(header_bytes)
        for row in rows:
            chunk = array(typecode, (cast(value) for value in row))
            if sys.byteorder != "little":
                chunk.byteswap()
            chunk.tofile(f)


def write_npy_vector(path: Path, values: list[Any], *, dtype: str) -> None:
    if dtype == "float32":
        descr = "<f4"
        typecode = "f"
        cast = float
    elif dtype == "int64":
        descr = "<i8"
        typecode = "q"
        cast = int
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    header_text = (
        "{'descr': '%s', 'fortran_order': False, 'shape': (%d,), }"
        % (descr, len(values))
    )
    header_bytes = header_text.encode("latin1")
    padding = 16 - ((10 + len(header_bytes) + 1) % 16)
    header_bytes += b" " * padding + b"\n"

    with path.open("wb") as f:
        f.write(b"\x93NUMPY")
        f.write(bytes([1, 0]))
        f.write(struct.pack("<H", len(header_bytes)))
        f.write(header_bytes)
        chunk = array(typecode, (cast(value) for value in values))
        if sys.byteorder != "little":
            chunk.byteswap()
        chunk.tofile(f)


def write_split_csv(path: Path, split: list[str], year_values: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row_index", "split", "year"])
        for idx, (split_name, year) in enumerate(zip(split, year_values)):
            writer.writerow([idx, split_name, year])


def write_row_metadata_csv(path: Path, rows: dict[str, list[str]]) -> None:
    fieldnames = ["row_index", "ID", "ID_fam", "year", "survey", "source", "sex", "age"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, year in enumerate(rows["__split_year__"]):
            writer.writerow(
                {
                    "row_index": idx,
                    "ID": rows["__id__"][idx],
                    "ID_fam": rows["__id_fam__"][idx],
                    "year": year,
                    "survey": rows["__survey__"][idx],
                    "source": rows["__source__"][idx],
                    "sex": rows["__sex__"][idx],
                    "age": rows["__age__"][idx],
                }
            )


def quantile_to_json(state: QuantileState) -> dict[str, Any]:
    return {
        "feature": state.feature,
        "n_valid_train": state.n_valid_train,
        "constant": state.constant,
        "quantiles": state.quantiles,
        "references": state.references,
        "normal_values": state.normal_values,
    }


if __name__ == "__main__":
    main()
