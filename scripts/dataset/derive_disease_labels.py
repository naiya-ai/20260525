"""Derive disease labels for harmonized KNHANES/NHANES CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .preprocess_variable_groups import (
        HarmonizedDatasetSpec,
        HarmonizedRowFilter,
        normalize_survey_name,
        parse_optional_float,
        parse_survey_filter,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from preprocess_variable_groups import (
        HarmonizedDatasetSpec,
        HarmonizedRowFilter,
        normalize_survey_name,
        parse_optional_float,
        parse_survey_filter,
    )


ID_COLUMNS = ["ID", "ID_fam", "year", "sex", "age"]


def main() -> None:
    args = parse_args()
    specs = resolve_input_specs(args)
    output_dir = Path(args.output_dir) if args.output_dir is not None else None

    for spec in specs:
        label_dir = output_dir or spec.path.parent
        output_path = label_dir / f"disease_labels_{spec.name}.csv"
        labels = derive_for_csv(spec.path, row_filter=spec.row_filter)
        label_dir.mkdir(parents=True, exist_ok=True)
        labels.to_csv(output_path, index=False)
        print(
            f"wrote {output_path} rows={len(labels)} "
            f"diseases={len([c for c in labels.columns if c.startswith('disease_')])}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harmonized-dataset", help="Single harmonized CSV path.")
    parser.add_argument("--dataset-name", help="Output dataset/view name for one CSV.")
    parser.add_argument(
        "--survey",
        nargs="+",
        help="Keep only these survey values when --harmonized-dataset is used.",
    )
    parser.add_argument("--year-min", type=float, help="Keep rows with year >= this value.")
    parser.add_argument("--year-max", type=float, help="Keep rows with year <= this value.")
    parser.add_argument("--datasets-dir", default="datasets/harmonized")
    parser.add_argument("--pattern", default="harmonized*.csv")
    parser.add_argument("--output-dir", help="Directory for disease label CSV files.")
    return parser.parse_args()


def resolve_input_specs(args: argparse.Namespace) -> list[HarmonizedDatasetSpec]:
    if args.harmonized_dataset is not None:
        path = Path(args.harmonized_dataset)
        row_filter = build_row_filter(args)
        return [
            HarmonizedDatasetSpec(
                path=path,
                name=args.dataset_name or path.stem,
                row_filter=row_filter,
            )
        ]

    datasets_dir = Path(args.datasets_dir)
    input_paths = sorted(datasets_dir.glob(args.pattern))
    if not input_paths:
        raise FileNotFoundError(
            f"No harmonized CSV files matched {args.pattern!r} in {datasets_dir}."
        )
    return [HarmonizedDatasetSpec(path=path, name=path.stem) for path in input_paths]


def build_row_filter(args: argparse.Namespace) -> HarmonizedRowFilter | None:
    surveys = parse_survey_filter(args.survey)
    year_min = parse_optional_float(args.year_min)
    year_max = parse_optional_float(args.year_max)
    if surveys is None and year_min is None and year_max is None:
        return None
    return HarmonizedRowFilter(surveys=surveys, year_min=year_min, year_max=year_max)


def derive_for_csv(
    path: Path,
    *,
    row_filter: HarmonizedRowFilter | None = None,
) -> pd.DataFrame:
    required_columns = sorted(
        set(ID_COLUMNS)
        | {
            "HE_glu",
            "HE_HbA1c",
            "HE_sbp",
            "HE_dbp",
            "HE_chol",
            "HE_TG",
            "HE_HDL_st2",
            "HE_ast",
            "HE_alt",
            "HE_hepaB",
            "HE_hepaC",
            "HE_crea",
            "HE_HB",
        }
    )
    if row_filter is not None and row_filter.surveys is not None:
        required_columns.append("survey")

    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
    missing = [column for column in required_columns if column not in header]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = pd.read_csv(
        path,
        usecols=required_columns,
        low_memory=False,
        encoding="utf-8-sig",
    )
    df = filter_harmonized_frame(df, row_filter)
    labels = df[[column for column in ID_COLUMNS if column in df.columns]].copy()
    labels["row_index"] = np.arange(len(df), dtype=np.int64)

    labels["disease_diabetes"] = _or_label(
        [
            df["HE_glu"].ge(126),
            df["HE_HbA1c"].ge(6.5),
        ],
        [
            df["HE_glu"].notna(),
            df["HE_HbA1c"].notna(),
        ],
    )
    labels["disease_hypertension"] = _or_label(
        [
            df["HE_sbp"].ge(140),
            df["HE_dbp"].ge(90),
        ],
        [
            df["HE_sbp"].notna(),
            df["HE_dbp"].notna(),
        ],
    )
    hdl_rule = (
        (df["sex"].eq(1) & df["HE_HDL_st2"].lt(40))
        | (df["sex"].eq(2) & df["HE_HDL_st2"].lt(50))
    )
    labels["disease_dyslipidemia"] = _or_label(
        [
            df["HE_chol"].ge(240),
            df["HE_TG"].ge(200),
            hdl_rule,
        ],
        [
            df["HE_chol"].notna(),
            df["HE_TG"].notna(),
            df["sex"].notna() & df["HE_HDL_st2"].notna(),
        ],
    )
    labels["disease_liver_disease"] = _or_label(
        [
            df["HE_ast"].ge(40),
            df["HE_alt"].ge(40),
        ],
        [
            df["HE_ast"].notna(),
            df["HE_alt"].notna(),
        ],
    )
    labels["disease_hepatitis_b"] = _single_condition_label(
        df["HE_hepaB"].eq(1),
        df["HE_hepaB"].notna(),
    )
    labels["disease_hepatitis_c"] = _single_condition_label(
        df["HE_hepaC"].eq(1),
        df["HE_hepaC"].notna(),
    )
    labels["disease_kidney_disease"] = _kidney_disease_label(df)
    labels["disease_anemia"] = _single_condition_label(
        (df["sex"].eq(1) & df["HE_HB"].lt(13))
        | (df["sex"].eq(2) & df["HE_HB"].lt(12)),
        df["sex"].notna() & df["HE_HB"].notna(),
    )

    disease_columns = [column for column in labels.columns if column.startswith("disease_")]
    return labels[[*ID_COLUMNS, "row_index", *disease_columns]]


def filter_harmonized_frame(
    df: pd.DataFrame,
    row_filter: HarmonizedRowFilter | None,
) -> pd.DataFrame:
    if row_filter is None:
        return df

    keep = pd.Series(True, index=df.index)
    if row_filter.surveys is not None:
        if "survey" not in df.columns:
            raise ValueError("Survey filtering requires a survey column.")
        survey = df["survey"].map(normalize_survey_name)
        keep &= survey.isin(row_filter.surveys)

    if row_filter.year_min is not None or row_filter.year_max is not None:
        year = pd.to_numeric(df["year"], errors="coerce")
        if row_filter.year_min is not None:
            keep &= year.ge(row_filter.year_min)
        if row_filter.year_max is not None:
            keep &= year.le(row_filter.year_max)

    filtered = df.loc[keep].reset_index(drop=True)
    if filtered.empty:
        raise ValueError(f"No rows remained after applying filter {row_filter}.")
    return filtered


def _or_label(
    conditions: list[pd.Series],
    observed: list[pd.Series],
) -> pd.Series:
    condition_frame = pd.concat(conditions, axis=1).fillna(False)
    observed_frame = pd.concat(observed, axis=1).fillna(False)
    true_mask = condition_frame.any(axis=1)
    false_mask = (~true_mask) & observed_frame.all(axis=1)
    return _nullable_boolean(true_mask, false_mask)


def _single_condition_label(condition: pd.Series, observed: pd.Series) -> pd.Series:
    true_mask = condition.fillna(False)
    false_mask = (~true_mask) & observed.fillna(False)
    return _nullable_boolean(true_mask, false_mask)


def _kidney_disease_label(df: pd.DataFrame) -> pd.Series:
    age = df["age"]
    sex = df["sex"]
    crea = df["HE_crea"]
    observed = age.notna() & sex.notna() & crea.notna()
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
    egfr = pd.Series(np.nan, index=df.index, dtype="float64")
    egfr = egfr.mask(sex.eq(2), female_egfr)
    egfr = egfr.mask(sex.eq(1), male_egfr)
    true_mask = age.gt(18) & egfr.lt(60)
    false_mask = observed & ~true_mask
    return _nullable_boolean(true_mask, false_mask)


def _nullable_boolean(true_mask: pd.Series, false_mask: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=true_mask.index, dtype="boolean")
    result[false_mask] = False
    result[true_mask] = True
    return result


if __name__ == "__main__":
    main()
