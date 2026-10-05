"""Validate the final EHR source and build deterministic experiment splits."""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ehr_contract import (  # noqa: E402
    AGE_COLUMN,
    CANONICAL_MODEL_COLUMNS,
    CANONICAL_SOURCE_COLUMNS,
    DISEASE_COLUMN,
    EXAMINATION_COLUMNS,
    HPO_COLUMN,
    IGNORED_IDENTIFIER_COLUMNS,
    PATIENT_IDENTIFIER_COLUMN,
    REQUIRED_SOURCE_COLUMNS,
    SEX_COLUMN,
    TABLE_COLUMN,
    ordered_active_examinations,
    ordered_model_columns,
)
from src.runtime_checks import check_writable_outputs  # noqa: E402


EHR_ROOT = ROOT / "datasets" / "ehr"
DEFAULT_RAW = EHR_ROOT / "raw" / "0813iei_final.csv"
DEFAULT_OUTPUT = EHR_ROOT
SPLIT_RANDOM_STATE = 0

# Canonical source fields; individual datasets may use a subset of examinations.
RETAINED_SOURCE_COLUMNS = CANONICAL_MODEL_COLUMNS
EXPECTED_SOURCE_COLUMNS = CANONICAL_SOURCE_COLUMNS
RECORD_ID_COLUMN = "record_id"
HPO_PATTERN = re.compile(r"HP:\d{7}")


def active_examination_columns(dataframe: pd.DataFrame) -> list[str]:
    """Treat absent or dataset-level all-missing examinations as inactive."""
    present = ordered_active_examinations(dataframe.columns.tolist())
    return [column for column in present if not dataframe[column].isna().all()]


def retained_source_columns(dataframe: pd.DataFrame) -> list[str]:
    return ordered_model_columns(active_examination_columns(dataframe))


def parse_table_labels(value: Any) -> list[str]:
    if pd.isna(value):
        raise ValueError(f"Missing {TABLE_COLUMN}")
    labels = [part.strip() for part in str(value).split(";") if part.strip()]
    if not labels:
        raise ValueError(f"Invalid {TABLE_COLUMN}: {value!r}")
    if len(labels) != len(set(labels)):
        raise ValueError(f"Duplicate labels in {TABLE_COLUMN}: {value!r}")
    return labels


def parse_hpo_codes(value: Any, field: str = HPO_COLUMN) -> list[str]:
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"Invalid {field}: {value!r}") from error
    if not isinstance(parsed, list):
        raise ValueError(f"{field} must be a list: {value!r}")
    codes: list[str] = []
    for code in parsed:
        if not isinstance(code, str) or HPO_PATTERN.fullmatch(code) is None:
            raise ValueError(f"Invalid HPO code in {field}: {code!r}")
        if code not in codes:
            codes.append(code)
    return sorted(codes)


def numeric(value: Any, field: str) -> int | float | None:
    if pd.isna(value) or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Non-numeric value for {field}: {value!r}") from error
    if not pd.notna(result) or result in {float("inf"), float("-inf")}:
        raise ValueError(f"Non-finite value for {field}: {value!r}")
    return int(result) if result.is_integer() else result


def load_raw(path: Path = DEFAULT_RAW) -> pd.DataFrame:
    dataframe = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={PATIENT_IDENTIFIER_COLUMN: str},
    )
    available = set(dataframe.columns)
    missing = sorted(set(REQUIRED_SOURCE_COLUMNS) - available)
    unknown = sorted(available - set(CANONICAL_SOURCE_COLUMNS))
    if missing or unknown:
        raise ValueError(
            "EHR source column contract mismatch: "
            f"missing_required={missing}, unknown={unknown}"
        )
    required = [TABLE_COLUMN, HPO_COLUMN, SEX_COLUMN, AGE_COLUMN]
    if dataframe.empty or dataframe[required].isna().any().any():
        raise ValueError("EHR source is empty or missing required values")
    invalid_sex = sorted(set(dataframe[SEX_COLUMN].astype(str)) - {"F", "M"})
    if invalid_sex:
        raise ValueError(f"Unsupported sex values: {invalid_sex}")
    disease_names = dataframe[DISEASE_COLUMN].fillna("").astype(str).str.strip()
    if disease_names.ne("").sum() == 0:
        raise ValueError("No non-empty disease names are available for generation")
    active = active_examination_columns(dataframe)
    if not active:
        raise ValueError("At least one non-empty examination column is required")
    return dataframe[ordered_model_columns(active)].copy()


def build_processed(dataframe: pd.DataFrame) -> pd.DataFrame:
    retained_columns = retained_source_columns(dataframe)
    examination_columns = active_examination_columns(dataframe)
    rows: list[pd.Series] = []
    record_ids: list[str] = []
    for source_index, (_, row) in enumerate(dataframe.iterrows(), start=1):
        labels = parse_table_labels(row[TABLE_COLUMN])
        for label_index, label in enumerate(labels, start=1):
            expanded = row.copy()
            expanded[TABLE_COLUMN] = label
            rows.append(expanded)
            suffix = f"_{label_index:02d}" if len(labels) > 1 else ""
            record_ids.append(f"ehr_{source_index:06d}{suffix}")
    processed = pd.DataFrame(rows).reset_index(drop=True)
    processed.insert(0, RECORD_ID_COLUMN, record_ids)
    processed[HPO_COLUMN] = [
        json.dumps(parse_hpo_codes(value), separators=(",", ":"))
        for value in processed[HPO_COLUMN]
    ]
    processed[DISEASE_COLUMN] = processed[DISEASE_COLUMN].map(
        lambda value: None if pd.isna(value) or not str(value).strip() else str(value).strip()
    )
    for column in [AGE_COLUMN, *examination_columns]:
        processed[column] = [numeric(value, column) for value in processed[column]]
    return processed[[RECORD_ID_COLUMN, *retained_columns]]


def expansion_source_indices(dataframe: pd.DataFrame) -> list[int]:
    return [
        source_index
        for source_index, value in enumerate(dataframe[TABLE_COLUMN])
        for _ in parse_table_labels(value)
    ]


def _select_train_groups(
    group_sizes: list[int],
    target_rows: int,
    train_fraction: float,
) -> set[int]:
    """Choose a deterministic non-empty proper subset nearest the row target."""
    if len(group_sizes) == 1:
        return {0}
    subsets: dict[int, tuple[int, ...]] = {0: ()}
    for group_index, size in enumerate(group_sizes):
        additions = {
            total + size: selected + (group_index,)
            for total, selected in list(subsets.items())
            if total + size not in subsets
        }
        subsets.update(additions)
    candidates = [
        (total, selected)
        for total, selected in subsets.items()
        if selected and len(selected) < len(group_sizes)
    ]
    _, selected = min(
        candidates,
        key=lambda item: (
            abs(item[0] - target_rows),
            abs(len(item[1]) / len(group_sizes) - train_fraction),
            item[1],
        ),
    )
    return set(selected)


def split_ids(
    dataframe: pd.DataFrame,
    train_fraction: float,
    patient_ids: list[str],
    stratification_labels: list[str] | None = None,
) -> dict[str, list[str]]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if len(patient_ids) != len(dataframe):
        raise ValueError("patient_ids must align one-to-one with EHR rows")
    if stratification_labels is None:
        stratification_labels = dataframe[TABLE_COLUMN].astype(str).tolist()
    if len(stratification_labels) != len(dataframe):
        raise ValueError("stratification_labels must align one-to-one with EHR rows")
    working = dataframe[[RECORD_ID_COLUMN, TABLE_COLUMN]].copy()
    working["_patient_id"] = [str(value) for value in patient_ids]
    working["_stratum"] = [str(value) for value in stratification_labels]
    labels_per_patient = working.groupby("_patient_id")["_stratum"].nunique()
    if (labels_per_patient > 1).any():
        raise ValueError("A patient cannot be split safely across multiple source strata")

    result = {"train": [], "test": []}
    for label_index, (_, label_rows) in enumerate(working.groupby("_stratum", sort=True)):
        groups = [
            (patient_id, group[RECORD_ID_COLUMN].tolist())
            for patient_id, group in label_rows.groupby("_patient_id", sort=True)
        ]
        random.Random(SPLIT_RANDOM_STATE * 1009 + label_index).shuffle(groups)
        selected = _select_train_groups(
            [len(record_ids) for _, record_ids in groups],
            int(round(len(label_rows) * train_fraction)),
            train_fraction,
        )
        for group_index, (_, record_ids) in enumerate(groups):
            destination = "train" if group_index in selected else "test"
            result[destination].extend(record_ids)
    for name in result:
        random.Random(SPLIT_RANDOM_STATE * 9173 + len(name)).shuffle(result[name])
    return result


def write_splits(
    dataframe: pd.DataFrame,
    split_dir: Path,
    ids: dict[str, list[str]],
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for stale in split_dir.iterdir():
        if stale.is_file() and stale.suffix in {".csv", ".json"}:
            stale.unlink()
    flattened = [record_id for values in ids.values() for record_id in values]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(
        dataframe[RECORD_ID_COLUMN]
    ):
        raise RuntimeError("Train/test IDs do not partition the dataset")
    indexed = dataframe.set_index(RECORD_ID_COLUMN, drop=False)
    for name, selected_ids in ids.items():
        indexed.loc[selected_ids].reset_index(drop=True).to_csv(
            split_dir / f"{name}.csv", index=False,
        )
    (split_dir / "split_ids.json").write_text(
        json.dumps(ids, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


def prepare_ehr(
    raw_path: Path = DEFAULT_RAW,
    output_dir: Path = DEFAULT_OUTPUT,
    train_fraction: float = 0.7,
) -> dict[str, Any]:
    check_writable_outputs(files=[
        output_dir / "processed" / "ehr.csv",
        output_dir / "splits" / "train.csv",
        output_dir / "splits" / "test.csv",
        output_dir / "splits" / "split_ids.json",
    ])
    raw = load_raw(raw_path)
    final = build_processed(raw)
    source = pd.read_csv(
        raw_path,
        encoding="utf-8-sig",
        dtype={PATIENT_IDENTIFIER_COLUMN: str},
    )
    source_indices = expansion_source_indices(raw)
    warnings: list[str] = []
    if PATIENT_IDENTIFIER_COLUMN in source.columns:
        identifiers = source[PATIENT_IDENTIFIER_COLUMN].fillna("").astype(str).str.strip()
        missing_identifiers = int(identifiers.eq("").sum())
        if missing_identifiers:
            raise ValueError(
                f"{PATIENT_IDENTIFIER_COLUMN} has {missing_identifiers} empty values"
            )
        patient_ids = identifiers.iloc[source_indices].tolist()
        patient_count = int(identifiers.nunique())
        split_grouping = PATIENT_IDENTIFIER_COLUMN
    else:
        patient_ids = [
            f"source_row_{source_index + 1:06d}"
            for source_index in source_indices
        ]
        patient_count = int(len(source))
        split_grouping = "source_row"
        warnings.append(
            f"{PATIENT_IDENTIFIER_COLUMN} is absent; assuming one source row per patient"
        )
    source_strata = source.iloc[source_indices][TABLE_COLUMN].astype(str).tolist()
    output_path = output_dir / "processed" / "ehr.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path, index=False)
    ids = split_ids(final, train_fraction, patient_ids, source_strata)
    write_splits(final, output_dir / "splits", ids)
    disease_tables = final.dropna(subset=[DISEASE_COLUMN]).groupby(
        DISEASE_COLUMN
    )[TABLE_COLUMN].nunique()
    active_examinations = active_examination_columns(source)
    multi_table_disease_names = sorted(
        disease_tables[disease_tables > 1].index.astype(str).tolist()
    )
    return {
        "rows": len(final),
        "columns": len(final.columns),
        "active_examination_columns": active_examinations,
        "inactive_examination_columns": [
            column
            for column in EXAMINATION_COLUMNS
            if column not in active_examinations
        ],
        "disease_names": int(final[DISEASE_COLUMN].nunique(dropna=True)),
        "missing_disease_names": int(final[DISEASE_COLUMN].isna().sum()),
        "multi_table_disease_names": multi_table_disease_names,
        "table_classes": int(final[TABLE_COLUMN].nunique()),
        "source_rows": len(raw),
        "patients": patient_count,
        "split_grouping": split_grouping,
        "split_rows": {name: len(values) for name, values in ids.items()},
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW,
                        help="EHR source CSV (default: datasets/ehr/raw/0813iei_final.csv)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Dataset directory receiving processed/ and splits/ (default: datasets/ehr)")
    parser.add_argument("--train-fraction", type=float, default=0.7,
                        help="Target training fraction for the fixed patient-grouped split (default: 0.7)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_ehr(
        raw_path=args.raw_csv,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
    )
    for warning in summary["warnings"]:
        print(f"WARNING: {warning}")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
