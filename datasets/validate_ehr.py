"""Validate an EHR raw CSV before running the canonical preparation pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_ehr  # noqa: E402


def _error(report: dict[str, Any], message: str) -> None:
    report["error_count"] += 1
    if len(report["errors"]) < 50:
        report["errors"].append(message)
    elif len(report["errors"]) == 50:
        report["errors"].append("Additional errors omitted; see error_count")


def _warning(report: dict[str, Any], message: str) -> None:
    report["warnings"].append(message)


def validate_raw_csv(path: Path) -> dict[str, Any]:
    """Return an aggregated schema and experiment-readiness report."""
    report: dict[str, Any] = {
        "path": str(path),
        "valid": False,
        "error_count": 0,
        "errors": [],
        "warnings": [],
        "statistics": {},
    }
    if not path.is_file():
        _error(report, f"CSV does not exist: {path}")
        return report
    try:
        dataframe = pd.read_csv(
            path,
            encoding="utf-8-sig",
            dtype={"patient_identifier": str},
        )
    except Exception as error:
        _error(report, f"CSV cannot be read: {type(error).__name__}: {error}")
        return report

    report["statistics"].update(
        {"source_rows": int(len(dataframe)), "source_columns": int(len(dataframe.columns))}
    )
    if dataframe.empty:
        _error(report, "CSV contains no data rows")
        return report

    actual = dataframe.columns.tolist()
    available = set(actual)
    missing = sorted(set(prepare_ehr.REQUIRED_SOURCE_COLUMNS) - available)
    extra = sorted(available - set(prepare_ehr.EXPECTED_SOURCE_COLUMNS))
    if missing or extra:
        _error(
            report,
            "Column contract mismatch. "
            f"missing_required={missing}, unknown={extra}",
        )

    present_examinations = prepare_ehr.ordered_active_examinations(actual)
    active_examinations = [
        column
        for column in present_examinations
        if not dataframe[column].isna().all()
    ]
    inactive_examinations = [
        column
        for column in prepare_ehr.EXAMINATION_COLUMNS
        if column not in active_examinations
    ]
    all_missing_examinations = [
        column
        for column in present_examinations
        if column not in active_examinations
    ]
    report["statistics"].update(
        {
            "active_examination_columns": active_examinations,
            "active_examination_count": len(active_examinations),
            "inactive_examination_columns": inactive_examinations,
        }
    )
    if not active_examinations:
        _error(report, "At least one non-empty examination column is required")
    if all_missing_examinations:
        _warning(
            report,
            "Dataset-level all-missing examination columns will be inactive: "
            f"{all_missing_examinations}",
        )

    patient_column = prepare_ehr.PATIENT_IDENTIFIER_COLUMN
    if patient_column in available:
        patients = dataframe[patient_column].fillna("").astype(str).str.strip()
        missing_patients = int(patients.eq("").sum())
        if missing_patients:
            _error(report, f"{patient_column} has {missing_patients} empty values")
        report["statistics"]["patients"] = int(patients[patients.ne("")].nunique())
        if report["statistics"]["patients"] < 2:
            _error(report, "At least two patients are required for a train/test split")
        report["statistics"]["split_grouping"] = patient_column
    else:
        report["statistics"]["patients"] = int(len(dataframe))
        report["statistics"]["split_grouping"] = "source_row"
        if len(dataframe) < 2:
            _error(report, "At least two source rows are required for a train/test split")
        _warning(
            report,
            f"{patient_column} is absent; preparation will assume one source row per patient",
        )

    required = [
        prepare_ehr.TABLE_COLUMN,
        prepare_ehr.HPO_COLUMN,
        prepare_ehr.SEX_COLUMN,
        prepare_ehr.AGE_COLUMN,
    ]
    for column in required:
        if column in available:
            missing_count = int(dataframe[column].isna().sum())
            if missing_count:
                _error(report, f"{column} has {missing_count} missing values")

    parsed_tables: list[list[str] | None] = []
    if prepare_ehr.TABLE_COLUMN in available:
        for row_number, value in enumerate(dataframe[prepare_ehr.TABLE_COLUMN], start=2):
            try:
                parsed_tables.append(prepare_ehr.parse_table_labels(value))
            except ValueError as error:
                parsed_tables.append(None)
                _error(report, f"row {row_number}: {error}")
        table_names = sorted(
            {label for labels in parsed_tables if labels for label in labels}
        )
        report["statistics"]["table_classes"] = len(table_names)
        report["statistics"]["expanded_rows"] = sum(
            len(labels) for labels in parsed_tables if labels
        )
        if len(table_names) < 2:
            _error(report, "At least two IUIS_table classes are required for AUC evaluation")

    if prepare_ehr.HPO_COLUMN in available:
        empty_hpo = 0
        for row_number, value in enumerate(dataframe[prepare_ehr.HPO_COLUMN], start=2):
            try:
                if not prepare_ehr.parse_hpo_codes(value):
                    empty_hpo += 1
            except ValueError as error:
                _error(report, f"row {row_number}: {error}")
        report["statistics"]["empty_hpo_rows"] = empty_hpo
        if empty_hpo:
            _warning(report, f"{empty_hpo} rows contain an empty HPO list; they will be retained")

    if prepare_ehr.SEX_COLUMN in available:
        invalid = sorted(
            set(dataframe[prepare_ehr.SEX_COLUMN].dropna().astype(str)) - {"F", "M"}
        )
        if invalid:
            _error(report, f"Unsupported sex values: {invalid}; allowed values are F and M")

    for column in [prepare_ehr.AGE_COLUMN, *prepare_ehr.EXAMINATION_COLUMNS]:
        if column not in available:
            continue
        for row_number, value in enumerate(dataframe[column], start=2):
            try:
                prepare_ehr.numeric(value, column)
            except ValueError as error:
                _error(report, f"row {row_number}: {error}")

    disease = prepare_ehr.DISEASE_COLUMN
    if disease in available:
        names = dataframe[disease].fillna("").astype(str).str.strip()
        missing_names = int(names.eq("").sum())
        report["statistics"].update(
            {
                "disease_names": int(names[names.ne("")].nunique()),
                "missing_disease_names": missing_names,
            }
        )
        if missing_names:
            _warning(
                report,
                f"{missing_names} source rows have no disease name; they remain missing and "
                "cannot be generation conditions",
            )
        if names.ne("").sum() == 0:
            _error(report, "No non-empty disease names are available for conditional generation")

    if (
        patient_column in available
        and prepare_ehr.TABLE_COLUMN in available
        and not any(labels is None for labels in parsed_tables)
    ):
        working = pd.DataFrame(
            {
                "patient": dataframe[patient_column].fillna("").astype(str),
                "stratum": dataframe[prepare_ehr.TABLE_COLUMN].astype(str),
            }
        )
        mixed = working.groupby("patient")["stratum"].nunique()
        mixed = mixed[mixed > 1]
        if not mixed.empty:
            _error(
                report,
                f"{len(mixed)} patients occur in multiple source IUIS_table strata; "
                "the deterministic grouped splitter cannot process them",
            )

    report["valid"] = report["error_count"] == 0
    return report


def print_report(report: dict[str, Any]) -> None:
    status = "PASS" if report["valid"] else "FAIL"
    print(f"EHR raw CSV validation: {status}")
    print(f"path: {report['path']}")
    print(f"error_count: {report['error_count']}")
    for key, value in sorted(report["statistics"].items()):
        print(f"{key}: {value}")
    for message in report["warnings"]:
        print(f"WARNING: {message}")
    for message in report["errors"]:
        print(f"ERROR: {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, required=True, help="EHR source CSV to validate before preparation")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_raw_csv(args.raw_csv)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_report(report)
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
