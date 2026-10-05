"""Dataset schemas and column interfaces for Curasyn."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.ehr_contract import (
    AGE_COLUMN,
    CANONICAL_MODEL_COLUMNS,
    DISEASE_COLUMN,
    EXAMINATION_COLUMNS,
    HPO_COLUMN,
    SEX_COLUMN,
    TABLE_COLUMN,
    ordered_active_examinations,
)


ROOT = Path(__file__).resolve().parents[1]
EHR_DATASET_PATHS = {
    "ehr": ROOT / "datasets" / "ehr" / "processed" / "ehr.csv",
    "private_ehr": ROOT / "datasets" / "private_ehr" / "processed" / "ehr.csv",
}
EHR_DATASET_PATH = EHR_DATASET_PATHS["ehr"]
SUPPORTED_DATASETS = tuple(EHR_DATASET_PATHS)
HPO_PATTERN = re.compile(r"HP:\d{7}")


def load_hpo_vocabulary_file(
    path: str | Path,
    dataset: str | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Load an ordered task vocabulary and its declared source."""
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError("A fixed HPO vocabulary file path is required")
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Configured HPO vocabulary does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("HPO vocabulary file must contain a JSON object")
    if type(payload.get("format_version")) is not int or payload["format_version"] != 1:
        raise ValueError("HPO vocabulary format_version must be 1")
    if payload.get("dataset") not in SUPPORTED_DATASETS:
        raise ValueError(f"HPO vocabulary dataset must be one of {SUPPORTED_DATASETS}")
    if dataset is not None and payload["dataset"] != dataset:
        raise ValueError(f"HPO vocabulary dataset must be {dataset!r}")
    for field in ("origin", "source_description", "hpo_ontology_version"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ValueError(f"HPO vocabulary {field} must be a non-empty string")
    codes = payload.get("codes")
    if not isinstance(codes, list) or not codes:
        raise ValueError("HPO vocabulary codes must be a non-empty array")
    for code in codes:
        if not isinstance(code, str) or HPO_PATTERN.fullmatch(code) is None:
            raise ValueError(f"Invalid HPO vocabulary code: {code!r}")
    if len(codes) != len(set(codes)):
        raise ValueError("HPO vocabulary codes must not contain duplicates")
    return list(codes), {
        "mode": "file",
        "path": str(path),
        "dataset": payload["dataset"],
        "origin": payload["origin"],
        "source_description": payload["source_description"],
        "hpo_ontology_version": payload["hpo_ontology_version"],
    }


def load_ehr_schema(
    dataset_path: str | Path = EHR_DATASET_PATH,
    *,
    hpo_vocabulary_path: str | Path | None = None,
    dataset: str = "ehr",
) -> dict[str, Any]:
    """Infer EHR fields and load the required fixed task vocabulary."""
    hpo_vocabulary, hpo_vocabulary_source = load_hpo_vocabulary_file(
        hpo_vocabulary_path, dataset,
    )
    dataset_path = Path(dataset_path)
    dataframe = pd.read_csv(dataset_path, keep_default_na=False)
    label_column = TABLE_COLUMN
    disease_column = DISEASE_COLUMN
    phenotype_column = HPO_COLUMN
    required = {
        "record_id",
        label_column,
        disease_column,
        phenotype_column,
        SEX_COLUMN,
        AGE_COLUMN,
    }
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"{dataset_path} is missing EHR columns: {missing}")

    demographic_columns = [SEX_COLUMN, AGE_COLUMN]
    reserved = {
        "record_id",
        label_column,
        disease_column,
        phenotype_column,
        *demographic_columns,
    }
    unexpected = sorted(
        set(dataframe.columns) - reserved - set(EXAMINATION_COLUMNS)
    )
    if unexpected:
        raise ValueError(
            f"{dataset_path} contains unknown processed EHR columns: {unexpected}"
        )
    examination_columns = ordered_active_examinations(dataframe.columns.tolist())
    if not examination_columns:
        raise ValueError("EHR examination columns are missing")

    demographic_fields = [
        {
            "raw_name": SEX_COLUMN,
            "semantic_name": SEX_COLUMN,
            "model_column": SEX_COLUMN,
            "value_type": "categorical",
            "mapping": {"F": 0, "M": 1},
        },
        {
            "raw_name": AGE_COLUMN,
            "semantic_name": AGE_COLUMN,
            "model_column": AGE_COLUMN,
            "value_type": "numeric",
            "unit": "years",
        },
    ]

    examination_fields = [
        {
            "raw_name": column,
            "semantic_name": column,
            "model_column": column,
            "value_type": "numeric",
            "unit": None,
        }
        for column in examination_columns
    ]
    feature_columns = [phenotype_column, *demographic_columns, *examination_columns]
    numerical_columns = [
        *[
            field["model_column"]
            for field in demographic_fields
            if field["value_type"] == "numeric"
        ],
        *examination_columns,
    ]
    categorical_columns = [
        phenotype_column,
        *[
            field["model_column"]
            for field in demographic_fields
            if field["value_type"] == "categorical"
        ],
    ]
    label_names = sorted(
        str(value).strip() for value in dataframe[label_column].unique() if str(value).strip()
    )
    if not label_names:
        raise ValueError("EHR has no IUIS Table labels")
    label_mapping = {name: index for index, name in enumerate(label_names)}
    return {
        "dataset": "ehr",
        "schema_version": 3,
        "record_id_column": "record_id",
        "metadata_columns": ["record_id"],
        "condition_columns": [disease_column],
        "disease_name_column": disease_column,
        "label_column": label_column,
        "label_mapping": label_mapping,
        "task_type": "multiclass",
        "n_classes": len(label_mapping),
        "feature_columns": feature_columns,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "phenotype_column": phenotype_column,
        "hpo_pattern": HPO_PATTERN.pattern,
        "hpo_vocabulary": hpo_vocabulary,
        "hpo_vocabulary_source": hpo_vocabulary_source,
        "canonical_model_columns": list(CANONICAL_MODEL_COLUMNS),
        "canonical_examination_columns": list(EXAMINATION_COLUMNS),
        "active_examination_columns": examination_columns,
        "inactive_examination_columns": [
            column for column in EXAMINATION_COLUMNS if column not in examination_columns
        ],
        "demographic_fields": demographic_fields,
        "examination_fields": examination_fields,
        "allow_missing_columns": examination_columns,
    }


def get_schema(
    dataset: str,
    dataset_path: str | Path | None = None,
    *,
    hpo_vocabulary_path: str | Path | None = None,
) -> dict[str, Any]:
    if dataset not in EHR_DATASET_PATHS:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Unsupported dataset {dataset!r}; supported: {supported}")
    dataset_path = Path(dataset_path).expanduser() if dataset_path else EHR_DATASET_PATHS[dataset]
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Prepared dataset is missing: {dataset_path}. "
            "Validate the raw CSV and run datasets/prepare_ehr.py first."
        )
    schema = load_ehr_schema(
        dataset_path, hpo_vocabulary_path=hpo_vocabulary_path, dataset=dataset,
    )
    schema["dataset_name"] = dataset
    return schema


def feature_columns(schema: dict[str, Any]) -> list[str]:
    return list(schema["feature_columns"])


def all_columns(schema: dict[str, Any]) -> list[str]:
    return feature_columns(schema) + [schema["label_column"]]


def metadata_columns(schema: dict[str, Any]) -> list[str]:
    return list(schema.get("metadata_columns", []))


def table_columns(
    schema: dict[str, Any],
    include_metadata: bool = False,
    include_conditions: bool = False,
) -> list[str]:
    columns = all_columns(schema)
    if include_conditions:
        columns = list(schema.get("condition_columns", [])) + columns
    if include_metadata:
        columns = metadata_columns(schema) + columns
    return columns
