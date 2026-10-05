"""Dataset loading and output cleaning for the Curasyn pipeline."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.schemas import SUPPORTED_DATASETS, all_columns, get_schema, table_columns


@dataclass(frozen=True)
class NumericalValidityProfile:
    """Train-only broad numerical envelopes used before learned curation."""

    bounds: dict[str, tuple[float, float] | None]

    @property
    def all_missing_columns(self) -> list[str]:
        return sorted(
            column for column, bounds in self.bounds.items() if bounds is None
        )


@dataclass(frozen=True)
class NumericalValidityPartition:
    valid_indices: list[int]
    rejected_indices: list[int]
    violation_counts: dict[str, int]


def build_numerical_validity_profile(
    train_df: pd.DataFrame,
    schema: dict[str, Any],
    *,
    iqr_multiplier: float = 3.0,
    zero_iqr_margin: float = 0.25,
) -> NumericalValidityProfile:
    """Freeze permissive, train-only ranges without imputing missing values."""
    if not np.isfinite(iqr_multiplier) or iqr_multiplier < 0:
        raise ValueError("curation.numerical_validity.iqr_multiplier must be finite and non-negative")
    if not np.isfinite(zero_iqr_margin) or zero_iqr_margin < 0:
        raise ValueError("curation.numerical_validity.zero_iqr_margin must be finite and non-negative")
    bounds: dict[str, tuple[float, float] | None] = {}
    for column in schema["numerical_columns"]:
        values = pd.to_numeric(train_df[column], errors="coerce").dropna()
        if values.empty:
            bounds[column] = None
            continue

        observed_min = float(values.min())
        observed_max = float(values.max())
        q1, q3 = np.quantile(values.to_numpy(dtype=float), [0.25, 0.75])
        iqr = float(q3 - q1)
        if iqr > 0:
            lower = min(
                observed_min,
                float(q1) - iqr_multiplier * iqr,
            )
            upper = max(
                observed_max,
                float(q3) + iqr_multiplier * iqr,
            )
        else:
            scale = max(abs(observed_min), abs(observed_max), 1.0)
            margin = zero_iqr_margin * scale
            lower = observed_min - margin
            upper = observed_max + margin
        bounds[column] = (float(lower), float(upper))
    return NumericalValidityProfile(bounds=bounds)


def partition_numerically_valid(
    dataframe: pd.DataFrame,
    profile: NumericalValidityProfile,
) -> NumericalValidityPartition:
    """Partition rows by obvious scale violations while preserving nulls."""
    valid_indices: list[int] = []
    rejected_indices: list[int] = []
    violation_counts: dict[str, int] = {}
    for position, (_, row) in enumerate(dataframe.iterrows()):
        violated: list[str] = []
        for column, bounds in profile.bounds.items():
            value = row[column]
            if pd.isna(value):
                continue
            if bounds is None:
                violated.append(column)
                continue
            numeric = float(value)
            lower, upper = bounds
            tolerance = 1e-12 * max(abs(lower), abs(upper), 1.0)
            if numeric < lower - tolerance or numeric > upper + tolerance:
                violated.append(column)
        if violated:
            rejected_indices.append(position)
            for column in violated:
                violation_counts[column] = violation_counts.get(column, 0) + 1
        else:
            valid_indices.append(position)
    return NumericalValidityPartition(
        valid_indices=valid_indices,
        rejected_indices=rejected_indices,
        violation_counts=violation_counts,
    )


def resolve_dataset_config(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config["data"]["dataset"]
    if dataset not in SUPPORTED_DATASETS:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Unsupported dataset {dataset!r}; supported: {supported}")
    data_seed = int(config["data"].get("data_seed", 0))
    paths = config.setdefault("paths", {})
    dataset_csv = str(Path(paths.get("dataset_csv") or f"datasets/{dataset}/processed/ehr.csv").expanduser())
    vocabulary_files = config["data"].get("hpo_vocabulary_files", {})
    if not isinstance(vocabulary_files, dict):
        raise ValueError("data.hpo_vocabulary_files must map dataset names to fixed vocabulary paths")
    unknown_datasets = set(vocabulary_files) - set(SUPPORTED_DATASETS)
    if unknown_datasets:
        raise ValueError(f"Unknown HPO vocabulary datasets: {sorted(unknown_datasets, key=str)}")
    vocabulary_path = vocabulary_files.get(dataset)
    if not isinstance(vocabulary_path, (str, Path)) or not str(vocabulary_path).strip():
        raise ValueError(
            f"Set data.hpo_vocabulary_files.{dataset} to a fixed HPO vocabulary file; "
            "CSV-based vocabulary inference is not supported"
        )
    schema = get_schema(
        dataset, dataset_path=dataset_csv, hpo_vocabulary_path=vocabulary_path,
    )
    run = config.setdefault("run", {})
    if run.get("seed") is None:
        run["seed"] = data_seed
    artifact_tag = run.get("artifact_tag")
    if artifact_tag is not None:
        artifact_tag = str(artifact_tag)
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", artifact_tag) is None:
            raise ValueError(
                "run.artifact_tag must contain only letters, numbers, '.', '_', "
                "or '-' and start with a letter or number"
            )

    config["data"]["processed_csv"] = dataset_csv
    paths["dataset_csv"] = dataset_csv
    split_name = "fixed70_30"
    final_target = int(config["generation"]["n_samples"])
    artifact_name = f"curasyn_s{int(run['seed'])}_n{final_target}"
    if artifact_tag:
        artifact_name = f"{artifact_name}_{artifact_tag}"
    config["artifact_name"] = artifact_name
    paths["train_csv"] = str(Path(paths.get("train_csv") or f"datasets/{dataset}/splits/train.csv").expanduser())
    paths["test_csv"] = str(Path(paths.get("test_csv") or f"datasets/{dataset}/splits/test.csv").expanduser())
    results_dir = Path(paths.get("results_dir") or f"results/{dataset}").expanduser()
    paths["results_dir"] = str(results_dir)
    paths["result_csv"] = str(results_dir / f"{artifact_name}_llm.csv")
    paths["effective_csv"] = str(results_dir / f"{artifact_name}_final.csv")
    paths["checkpoint_state"] = str(results_dir / f"{artifact_name}_checkpoint.json")
    paths["checkpoint_journal"] = str(results_dir / f"{artifact_name}_checkpoint.jsonl")

    config["data"]["split_name"] = split_name
    config["schema"] = schema
    return config


def load_split(
    path: str | Path,
    schema: dict[str, Any],
    include_metadata: bool = False,
    include_conditions: bool = False,
) -> pd.DataFrame:
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    if schema.get("dataset") == "ehr":
        df = normalize_ehr_frame(df, schema)
    columns = table_columns(
        schema,
        include_metadata=include_metadata,
        include_conditions=include_conditions,
    )
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")
    return df[columns].copy()


def parse_hpo_codes(value: Any, schema: dict[str, Any]) -> list[str]:
    if schema.get("dataset") != "ehr":
        raise ValueError("HPO parsing is only defined for the EHR schema")
    if isinstance(value, list):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid HPO JSON array: {value!r}") from error
    else:
        raise ValueError(f"HPO value must be a list or JSON array: {value!r}")
    if not isinstance(parsed, list):
        raise ValueError(f"HPO value must decode to a list: {value!r}")

    pattern = re.compile(str(schema["hpo_pattern"]))
    vocabulary = set(schema["hpo_vocabulary"])
    codes: list[str] = []
    for code in parsed:
        if not isinstance(code, str) or pattern.fullmatch(code) is None:
            raise ValueError(f"Invalid HPO code: {code!r}")
        if code not in vocabulary:
            raise ValueError(f"HPO code is not in the frozen vocabulary: {code!r}")
        if code not in codes:
            codes.append(code)
    return sorted(codes)


def serialize_hpo_codes(value: Any, schema: dict[str, Any]) -> str:
    codes = parse_hpo_codes(value, schema)
    return json.dumps(codes, separators=(",", ":"), ensure_ascii=True)


def _semantic_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _decode_field_value(value: Any, field: dict[str, Any]) -> Any:
    value = _semantic_scalar(value)
    if value is None or field["value_type"] != "categorical":
        return value
    mapping = field.get("mapping")
    if not mapping:
        return value
    inverse = {encoded: semantic for semantic, encoded in mapping.items()}
    if value not in inverse:
        raise ValueError(
            f"Unknown encoded value for {field['semantic_name']}: {value!r}"
        )
    return inverse[value]


def _encode_field_value(value: Any, field: dict[str, Any]) -> Any:
    if value is None or field["value_type"] != "categorical":
        return value
    mapping = field.get("mapping")
    if not mapping:
        return value
    if value in mapping:
        return mapping[value]
    if value in mapping.values():
        return value
    raise ValueError(f"Unknown value for {field['semantic_name']}: {value!r}")


def normalize_ehr_frame(
    dataframe: pd.DataFrame,
    schema: dict[str, Any],
) -> pd.DataFrame:
    """Encode persisted EHR labels and categorical values for model use."""
    if schema.get("dataset") != "ehr":
        return dataframe.copy()
    normalized = dataframe.copy()
    label = schema["label_column"]
    if label in normalized:
        label_ids = set(int(value) for value in schema["label_mapping"].values())
        numeric_labels = pd.to_numeric(normalized[label], errors="coerce")
        already_encoded = (
            numeric_labels.notna().all()
            and np.equal(numeric_labels, np.floor(numeric_labels)).all()
            and set(numeric_labels.astype(int)).issubset(label_ids)
        )
        if already_encoded:
            normalized[label] = numeric_labels.astype(int)
        else:
            mapped = normalized[label].map(schema["label_mapping"])
            if mapped.isna().any():
                unknown = sorted(normalized.loc[mapped.isna(), label].astype(str).unique())
                raise ValueError(f"Unknown EHR IUIS Table labels: {unknown}")
            normalized[label] = mapped.astype(int)
    for field in schema["demographic_fields"]:
        column = field["model_column"]
        if field["value_type"] == "categorical" and column in normalized:
            normalized[column] = [
                _encode_field_value(value, field) for value in normalized[column]
            ]
    return normalized


def dataframe_to_pid_records(
    dataframe: pd.DataFrame,
    schema: dict[str, Any],
) -> list[dict[str, Any]]:
    if schema.get("dataset") != "ehr":
        raise ValueError("PID semantic records require the EHR schema")
    required = table_columns(schema)
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"EHR dataframe is missing required columns: {missing}")

    record_id_column = schema["record_id_column"]
    label_column = schema["label_column"]
    inverse_labels = {
        int(label_id): label_name
        for label_name, label_id in schema["label_mapping"].items()
    }
    records: list[dict[str, Any]] = []
    for _, row in dataframe.iterrows():
        label = int(row[label_column])
        if label not in inverse_labels:
            raise ValueError(f"Unknown EHR disease label: {label}")
        record_id = None
        if record_id_column in dataframe.columns and pd.notna(row[record_id_column]):
            record_id = str(row[record_id_column])
        records.append(
            {
                "record_id": record_id,
                "y": label,
                "y_name": inverse_labels[label],
                "P": parse_hpo_codes(row[schema["phenotype_column"]], schema),
                "E": {
                    field["semantic_name"]: _decode_field_value(
                        row[field["model_column"]],
                        field,
                    )
                    for field in schema["examination_fields"]
                },
                "G": {
                    field["semantic_name"]: _decode_field_value(
                        row[field["model_column"]],
                        field,
                    )
                    for field in schema["demographic_fields"]
                },
            }
        )
    return records


def pid_records_to_dataframe(
    records: list[dict[str, Any]],
    schema: dict[str, Any],
    include_metadata: bool = True,
) -> pd.DataFrame:
    if schema.get("dataset") != "ehr":
        raise ValueError("PID semantic records require the EHR schema")
    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {
            schema["label_column"]: int(record["y"]),
            schema["phenotype_column"]: serialize_hpo_codes(record["P"], schema),
        }
        if include_metadata:
            record_id = record.get("record_id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError("A non-empty record_id is required with metadata")
            row[schema["record_id_column"]] = record_id
        for field in schema["demographic_fields"]:
            row[field["model_column"]] = _encode_field_value(
                record["G"].get(field["semantic_name"]),
                field,
            )
        for field in schema["examination_fields"]:
            row[field["model_column"]] = _encode_field_value(
                record["E"].get(field["semantic_name"]),
                field,
            )
        rows.append(row)
    return pd.DataFrame(
        rows,
        columns=table_columns(schema, include_metadata=include_metadata),
    )


def pid_record_to_generation_fields(
    record: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        schema["phenotype_column"]: list(record["P"]),
    }
    for field in schema["demographic_fields"]:
        fields[field["model_column"]] = record["G"].get(field["semantic_name"])
    for field in schema["examination_fields"]:
        fields[field["model_column"]] = record["E"].get(field["semantic_name"])
    return fields


def normalize_pid_generated(
    records: list[dict[str, Any]],
    schema: dict[str, Any],
    target_label: int,
) -> pd.DataFrame:
    if schema.get("dataset") != "ehr":
        raise ValueError("PID generated-row normalization requires the EHR schema")
    if target_label not in set(schema["label_mapping"].values()):
        raise ValueError(f"Unknown target disease label: {target_label}")

    allowed_missing = set(schema.get("allow_missing_columns", []))
    normalized: list[dict[str, Any]] = []
    for record in records:
        try:
            row: dict[str, Any] = {
                schema["phenotype_column"]: serialize_hpo_codes(
                    record[schema["phenotype_column"]],
                    schema,
                ),
                schema["label_column"]: int(target_label),
            }
            valid = True
            for field in [
                *schema["demographic_fields"],
                *schema["examination_fields"],
            ]:
                column = field["model_column"]
                value = record.get(column)
                if value is None:
                    if column not in allowed_missing:
                        valid = False
                        break
                    row[column] = None
                    continue
                if field["value_type"] == "categorical":
                    row[column] = _encode_field_value(value, field)
                elif field["value_type"] == "numeric":
                    numeric = float(value)
                    if not pd.notna(numeric) or numeric in {float("inf"), float("-inf")}:
                        valid = False
                        break
                    row[column] = int(numeric) if numeric.is_integer() else numeric
                else:
                    valid = False
                    break
            if valid:
                normalized.append(row)
        except (KeyError, TypeError, ValueError):
            continue
    return pd.DataFrame(normalized, columns=all_columns(schema))


def clean_generated(
    df: pd.DataFrame,
    reference: pd.DataFrame,
    schema: dict[str, Any],
) -> pd.DataFrame:
    columns = all_columns(schema)
    df = df.copy()
    label = schema["label_column"]
    if "target" in df.columns:
        if label not in df.columns:
            df[label] = df["target"]
        else:
            df[label] = df[label].where(df[label].notna(), df["target"])
    missing = [column for column in columns if column not in df.columns]
    if missing:
        return pd.DataFrame(columns=columns).astype(reference.dtypes.to_dict())

    has_list = df[columns].map(lambda value: isinstance(value, list)).any(axis=1)
    df = df.loc[~has_list, columns].dropna(how="any")
    for column in schema["numerical_columns"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in schema["categorical_columns"] + [schema["label_column"]]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(how="any")

    for column in columns:
        try:
            df[column] = df[column].astype(reference[column].dtype)
        except (TypeError, ValueError):
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(how="any")[columns].reset_index(drop=True)
