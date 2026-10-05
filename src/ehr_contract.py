"""Canonical source-field contract shared by EHR preparation and runtime schema."""

from __future__ import annotations


PATIENT_IDENTIFIER_COLUMN = "patient_identifier"
IGNORED_IDENTIFIER_COLUMNS = [
    PATIENT_IDENTIFIER_COLUMN,
    "hospital_admission_identifier",
    "discharge_note_identifier",
]

TABLE_COLUMN = "IUIS_table"
DISEASE_COLUMN = "inborn_error_of_immunity_diagnosis"
HPO_COLUMN = "hpo_codes"
SEX_COLUMN = "sex"
AGE_COLUMN = "age_in_years"

EXAMINATION_COLUMNS = [
    "white_blood_cell_count",
    "hemoglobin_concentration",
    "platelet_count",
    "red_blood_cell_count",
    "mean_corpuscular_volume",
    "mean_corpuscular_hemoglobin",
    "mean_corpuscular_hemoglobin_concentration",
    "red_cell_distribution_width",
    "neutrophil_percentage",
    "lymphocyte_percentage",
    "monocyte_percentage",
    "absolute_lymphocyte_count",
    "absolute_neutrophil_count",
    "serum_sodium",
    "serum_potassium",
    "serum_chloride",
    "serum_glucose",
    "serum_creatinine",
    "blood_urea_nitrogen",
    "total_calcium",
    "serum_magnesium",
    "serum_phosphate",
    "alanine_aminotransferase",
    "aspartate_aminotransferase",
    "alkaline_phosphatase",
    "total_bilirubin",
    "serum_albumin",
    "prothrombin_time",
    "activated_partial_thromboplastin_time",
    "immunoglobulin_g",
    "immunoglobulin_m",
]

REQUIRED_SOURCE_COLUMNS = [
    DISEASE_COLUMN,
    HPO_COLUMN,
    SEX_COLUMN,
    AGE_COLUMN,
    TABLE_COLUMN,
]
CANONICAL_MODEL_COLUMNS = [
    DISEASE_COLUMN,
    HPO_COLUMN,
    SEX_COLUMN,
    AGE_COLUMN,
    *EXAMINATION_COLUMNS,
    TABLE_COLUMN,
]
CANONICAL_SOURCE_COLUMNS = [
    *IGNORED_IDENTIFIER_COLUMNS,
    *CANONICAL_MODEL_COLUMNS,
]


def ordered_active_examinations(columns: list[str] | tuple[str, ...]) -> list[str]:
    """Return present examination fields in canonical public-EHR order."""
    present = set(columns)
    return [column for column in EXAMINATION_COLUMNS if column in present]


def ordered_model_columns(examination_columns: list[str]) -> list[str]:
    """Build the canonical runtime order for one EHR dataset instance."""
    active = set(examination_columns)
    return [
        DISEASE_COLUMN,
        HPO_COLUMN,
        SEX_COLUMN,
        AGE_COLUMN,
        *[column for column in EXAMINATION_COLUMNS if column in active],
        TABLE_COLUMN,
    ]
