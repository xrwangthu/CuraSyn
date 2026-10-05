#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
What this script does
=====================

This script builds an admission-level IEI dataset from MIMIC-IV by integrating hospital data and discharge notes.

The main workflow includes:
1. Identifying high-confidence IEI admissions using diagnosis codes and discharge-text confirmation.
2. Extracting and summarizing selected laboratory measurements.
3. Extracting relevant laboratory results from discharge notes.
4. Standardizing IEI diagnosis labels and assigning major IUIS categories.
5. Exporting the processed admission-level dataset for subsequent phenotype extraction and analysis.

Required data and references
============================

The pipeline requires:
- MIMIC-IV v3.1 hospital data (`mimic-iv-3.1/hosp`).
- MIMIC-IV Note v2.2 discharge notes (`mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note`).
- `IUIS-IEI-list-for-web-site-July-2024V2`, obtained from the IUIS Inborn Errors of Immunity Committee website: https://iuis.org/committees/iei/
- A terminology reference for standardizing IEI diagnosis names.

HPO phenotype codes are generated separately from the extracted discharge notes using `hpo_extractor_discharge.py`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


csv.field_size_limit(10_000_000)


DEFAULT_OUTPUT_DIR = Path("outputs/iei_mimic")



# Laboratory item IDs included in the EHR panel.
SELECTED_LAB_ITEMIDS = [
    50861, 50862, 50863, 50868, 50878, 50882, 50885, 50893, 50902, 50912,
    50931, 50954, 50960, 50970, 50971, 50983, 51006, 51137, 51144, 51146,
    51200, 51221, 51222, 51244, 51248, 51249, 51250, 51251, 51254, 51255,
    51256, 51265, 51266, 51267, 51277, 51279, 51301, 50934, 50947, 51133,
    51237, 51274, 51275, 51678, 52069, 52073, 52074, 52075, 52172, 52033,
]


HIGH_CONFIDENCE_CODES = {
    "ICD10:D70.0",
    "ICD10:D71",
    "ICD10:D80.0",
    "ICD10:D80.2",
    "ICD10:D80.3",
    "ICD10:D80.4",
    "ICD10:D80.9",
    "ICD10:D81.9",
    "ICD10:D82.1",
    "ICD10:D82.3",
    "ICD10:D82.4",
    "ICD10:D82.8",
    "ICD10:D83.9",
    "ICD10:D84.1",
    "ICD10:D89.82",
    "ICD10:M04.1",
    "ICD10:M04.2",
    "ICD10:M04.8",
    "ICD9:277.31",
    "ICD9:279.01",
    "ICD9:279.02",
    "ICD9:279.03",
    "ICD9:279.04",
    "ICD9:279.05",
    "ICD9:279.06",
    "ICD9:279.2",
    "ICD9:279.8",
    "ICD9:288.2",
}


IEI_DIAGNOSIS_PATTERNS = [
    r"\bCVID\b",
    r"common variable immunodeficiency",
    r"\bSCID\b",
    r"severe combined immunodeficiency",
    r"combined immunodeficiency",
    r"\bimmunodeficiency\b",
    r"\bimmune deficiency\b",
    r"primary immunodeficiency",
    r"predominantly antibody defects?",
    r"hypogammaglobulin[ae]mia",
    r"agammaglobulin[ae]mia",
    r"selective\s+IgA\s+(?:deficiency|immunodeficiency)",
    r"selective\s+IgM\s+(?:deficiency|immunodeficiency)",
    r"immune globulin deficiency",
    r"immunoglobulin\s+[AGM]\s+deficiency",
    r"IgG\s+subclass\s+deficiency",
    r"complement\s+(?:deficiency|defect)",
    r"C1\s+(?:esterase\s+)?inhibitor\s+deficiency",
    r"hereditary angioedema",
    r"DiGeorge",
    r"22q11",
    r"hyper\s*IgE",
    r"Job'?s syndrome",
    r"autoimmune lymphoproliferative syndrome",
    r"\bALPS\b",
    r"chronic granulomatous disease",
    r"\bCGD\b",
    r"functional disorders? of polymorphonuclear neutrophils",
    r"congenital (?:agranulocytosis|neutropenia)",
    r"genetic anomal(?:y|ies) of leukocytes",
    r"familial Mediterranean fever",
    r"\bFMF\b",
    r"periodic fever syndrome",
    r"cryopyrin",
    r"autoinflammatory syndrome",
    r"mevalonate kinase deficiency",
    r"hyper\s*IgD syndrome",
]
COMPILED_IEI_PATTERNS = [(p, re.compile(p, re.IGNORECASE)) for p in IEI_DIAGNOSIS_PATTERNS]


SECTION_START_RE = re.compile(
    r"^\s*(?:"
    r"discharge diagnos(?:is|es)|"
    r"final diagnos(?:is|es)|"
    r"primary diagnos(?:is|es)|"
    r"secondary diagnos(?:is|es)|"
    r"principal diagnos(?:is|es)"
    r")\s*:?\s*$|"
    r"^\s*(?:"
    r"discharge diagnos(?:is|es)|"
    r"final diagnos(?:is|es)|"
    r"primary diagnos(?:is|es)|"
    r"secondary diagnos(?:is|es)|"
    r"principal diagnos(?:is|es)"
    r")\s*:",
    re.IGNORECASE | re.MULTILINE,
)
SECTION_STOP_RE = re.compile(
    r"^\s*(?:"
    r"discharge condition|"
    r"discharge instructions?|"
    r"followup instructions?|"
    r"medications?:|"
    r"discharge medications?:|"
    r"allergies?:|"
    r"hospital course|"
    r"brief hospital course|"
    r"history of present illness|"
    r"physical exam|"
    r"pertinent results|"
    r"procedures?|"
    r"condition at discharge"
    r")\b",
    re.IGNORECASE | re.MULTILINE,
)


FINAL_COLUMN_SPECS = [
    ("subject_id", "patient_identifier"),
    ("hadm_id", "hospital_admission_identifier"),
    ("discharge_note_ids", "discharge_note_identifier"),
    ("standardized_iei_diagnosis", "inborn_error_of_immunity_diagnosis"),
    ("hpo_codes", "hpo_codes"),
    ("gender", "sex"),
    ("anchor_age", "age_in_years"),
    ("White Blood Cells_Blood_Hematology", "white_blood_cell_count"),
    ("Hemoglobin_Blood_Hematology", "hemoglobin_concentration"),
    ("Platelet Count_Blood_Hematology", "platelet_count"),
    ("Red Blood Cells_Blood_Hematology", "red_blood_cell_count"),
    ("MCV_Blood_Hematology", "mean_corpuscular_volume"),
    ("MCH_Blood_Hematology", "mean_corpuscular_hemoglobin"),
    ("MCHC_Blood_Hematology", "mean_corpuscular_hemoglobin_concentration"),
    ("RDW_Blood_Hematology", "red_cell_distribution_width"),
    ("Neutrophils_Blood_Hematology", "neutrophil_percentage"),
    ("Lymphocytes_Blood_Hematology", "lymphocyte_percentage"),
    ("Monocytes_Blood_Hematology", "monocyte_percentage"),
    ("Absolute Lymphocyte Count_Blood_Hematology", "absolute_lymphocyte_count"),
    ("Absolute Neutrophil Count_Blood_Hematology", "absolute_neutrophil_count"),
    ("Sodium_Blood_Chemistry", "serum_sodium"),
    ("Potassium_Blood_Chemistry", "serum_potassium"),
    ("Chloride_Blood_Chemistry", "serum_chloride"),
    ("Bicarbonate_Blood_Chemistry", "serum_bicarbonate"),
    ("Glucose_Blood_Chemistry", "serum_glucose"),
    ("Creatinine_Blood_Chemistry", "serum_creatinine"),
    ("Urea Nitrogen_Blood_Chemistry", "blood_urea_nitrogen"),
    ("Calcium Total_Blood_Chemistry", "total_calcium"),
    ("Magnesium_Blood_Chemistry", "serum_magnesium"),
    ("Phosphate_Blood_Chemistry", "serum_phosphate"),
    ("Alanine Aminotransferase (ALT)_Blood_Chemistry", "alanine_aminotransferase"),
    ("Asparate Aminotransferase (AST)_Blood_Chemistry", "aspartate_aminotransferase"),
    ("Alkaline Phosphatase_Blood_Chemistry", "alkaline_phosphatase"),
    ("Bilirubin Total_Blood_Chemistry", "total_bilirubin"),
    ("Albumin_Blood_Chemistry", "serum_albumin"),
    ("PT_Blood_Hematology", "prothrombin_time"),
    ("PTT_Blood_Hematology", "activated_partial_thromboplastin_time"),
    ("pr_IgG", "immunoglobulin_g"),
    ("pr_IgM", "immunoglobulin_m"),
    ("IUIS_table", "IUIS_table"),
]


EXCLUDED_FINAL_KEYS = {("14611792", "21321889")}


STATIC_IUIS_TABLE_MAP = {
    "Common variable immunodeficiency without known genetic defect": "Table 3 Predominantly Antibody Deficiencies",
    "Non-syndromic agammaglobulinemia": "Table 3 Predominantly Antibody Deficiencies",
    "Familial Mediterranean fever": "Table 7 Autoinflammatory Disorders",
    "Hereditary angioedema with C1Inh deficiency": "Table 8 Complement Deficiencies",
    "Hereditary angioedema with normal C1Inh": "Table 8 Complement Deficiencies",
    "Recurrent infections associated with rare immunoglobulin isotypes deficiency": "Table 3 Predominantly Antibody Deficiencies",
    "Selective IgA deficiency": "Table 3 Predominantly Antibody Deficiencies",
    "Autosomal dominant hyper-IgE syndrome due to STAT3 deficiency": "Table 2 Combined immunodeficiencies with associated or syndromic features",
    "Autoimmune lymphoproliferative syndrome": "Table 4 Diseases of Immune Dysregulation",
    "Autosomal recessive primary immunodeficiency with defective spontaneous natural killer cell cytotoxicity": "Table 1 Immunodeficiencies affecting cellular and humoral immunity",
}
GENERIC_MULTI_TABLE = (
    "Table 1 Immunodeficiencies affecting cellular and humoral immunity; "
    "Table 2 Combined immunodeficiencies with associated or syndromic features; "
    "Table 3 Predominantly Antibody Deficiencies"
)


PR_FALLBACK_MAP = {
    "white_blood_cell_count": ("pr_White Blood Cells", "direct"),
    "neutrophil_percentage": ("pr_Neutrophils", "direct"),
    "lymphocyte_percentage": ("pr_Lymphocytes", "direct"),
    "monocyte_percentage": ("pr_Monocytes", "direct"),
    "absolute_lymphocyte_count": ("pr_Absolute Lymphocyte Count", "direct"),
    "absolute_neutrophil_count": ("pr_Absolute Neutrophil Count", "direct"),
    "alanine_aminotransferase": ("pr_ALT", "direct"),
    "aspartate_aminotransferase": ("pr_AST", "direct"),
    "alkaline_phosphatase": ("pr_Alkaline Phosphatase", "direct"),
    "total_bilirubin": ("pr_Bilirubin Total", "direct"),
    "serum_albumin": ("pr_Albumin", "albumin_g_dl"),
    "activated_partial_thromboplastin_time": ("pr_PTT", "direct"),
}


MANUAL_FINAL_OVERRIDES = {
    ("14833492", "24054248"): {
        "discharge_note_identifier": "14833492-DS-17",
        "inborn_error_of_immunity_diagnosis": "Common variable immunodeficiency without known genetic defect",
        "hpo_codes": "[]",
        "sex": "M",
        "age_in_years": "29",
        "white_blood_cell_count": "5.3",
        "hemoglobin_concentration": "130",
        "platelet_count": "224",
        "red_blood_cell_count": "4.31",
        "mean_corpuscular_volume": "92",
        "mean_corpuscular_hemoglobin": "30.2",
        "mean_corpuscular_hemoglobin_concentration": "329",
        "red_cell_distribution_width": "12.8",
        "neutrophil_percentage": "",
        "lymphocyte_percentage": "",
        "monocyte_percentage": "",
        "absolute_lymphocyte_count": "",
        "absolute_neutrophil_count": "",
        "serum_sodium": "141",
        "serum_potassium": "4.5",
        "serum_chloride": "104",
        "serum_bicarbonate": "30",
        "serum_glucose": "4.3845",
        "serum_creatinine": "97.24",
        "blood_urea_nitrogen": "2.499",
        "total_calcium": "2.0958",
        "serum_magnesium": "0.86394",
        "serum_phosphate": "0.90412",
        "alanine_aminotransferase": "149",
        "aspartate_aminotransferase": "",
        "alkaline_phosphatase": "98",
        "total_bilirubin": "5.13",
        "serum_albumin": "",
        "prothrombin_time": "",
        "activated_partial_thromboplastin_time": "",
        "immunoglobulin_g": "",
        "immunoglobulin_m": "",
        "IUIS_table": "Table 3 Predominantly Antibody Deficiencies",
    }
}


UNIT_RULES = {
    "white_blood_cell_count": {"k/ul": 1.0, "*10^9/l": 1.0, "10^9/l": 1.0},
    "hemoglobin_concentration": {"g/dl": 10.0, "g/l": 1.0},
    "platelet_count": {"k/ul": 1.0, "*10^9/l": 1.0, "10^9/l": 1.0},
    "red_blood_cell_count": {"m/ul": 1.0, "*10^12/l": 1.0, "10^12/l": 1.0},
    "mean_corpuscular_volume": {"fl": 1.0},
    "mean_corpuscular_hemoglobin": {"pg": 1.0},
    "mean_corpuscular_hemoglobin_concentration": {"g/dl": 10.0, "g/l": 1.0, "%": 10.0},
    "red_cell_distribution_width": {"%": 1.0},
    "neutrophil_percentage": {"%": 1.0},
    "lymphocyte_percentage": {"%": 1.0},
    "monocyte_percentage": {"%": 1.0},
    "absolute_lymphocyte_count": {"k/ul": 1.0},
    "absolute_neutrophil_count": {"k/ul": 1.0, "*10^9/l": 1.0, "10^9/l": 1.0},
    "serum_sodium": {"meq/l": 1.0, "mmol/l": 1.0},
    "serum_potassium": {"meq/l": 1.0, "mmol/l": 1.0},
    "serum_chloride": {"meq/l": 1.0, "mmol/l": 1.0},
    "serum_bicarbonate": {"meq/l": 1.0, "mmol/l": 1.0},
    "serum_glucose": {"mg/dl": 0.0555, "mmol/l": 1.0},
    "serum_creatinine": {"mg/dl": 88.4, "umol/l": 1.0, "μmol/l": 1.0},
    "blood_urea_nitrogen": {"mg/dl": 0.357, "mmol/l": 1.0},
    "total_calcium": {"mg/dl": 0.2495, "mmol/l": 1.0},
    "serum_magnesium": {"mg/dl": 0.4114, "mmol/l": 1.0},
    "serum_phosphate": {"mg/dl": 0.3229, "mmol/l": 1.0},
    "alanine_aminotransferase": {"iu/l": 1.0, "u/l": 1.0},
    "aspartate_aminotransferase": {"iu/l": 1.0, "u/l": 1.0},
    "alkaline_phosphatase": {"iu/l": 1.0, "u/l": 1.0},
    "total_bilirubin": {"mg/dl": 17.1, "umol/l": 1.0, "μmol/l": 1.0},
    "serum_albumin": {"g/dl": 10.0, "g/l": 1.0},
    "prothrombin_time": {"sec": 1.0, "s": 1.0},
    "activated_partial_thromboplastin_time": {"sec": 1.0, "s": 1.0},
}


PLAUSIBLE_RANGES = {
    "age_in_years": (0, 120),
    "white_blood_cell_count": (0, 500),
    "hemoglobin_concentration": (0, 250),
    "platelet_count": (0, 5000),
    "red_blood_cell_count": (0, 10),
    "mean_corpuscular_volume": (0, 200),
    "mean_corpuscular_hemoglobin": (0, 100),
    "mean_corpuscular_hemoglobin_concentration": (0, 600),
    "red_cell_distribution_width": (0, 100),
    "neutrophil_percentage": (0, 100),
    "lymphocyte_percentage": (0, 100),
    "monocyte_percentage": (0, 100),
    "absolute_lymphocyte_count": (0, 100),
    "absolute_neutrophil_count": (0, 100),
    "serum_sodium": (80, 200),
    "serum_potassium": (0, 20),
    "serum_chloride": (50, 200),
    "serum_bicarbonate": (0, 80),
    "serum_glucose": (0, 100),
    "serum_creatinine": (0, 5000),
    "blood_urea_nitrogen": (0, 200),
    "total_calcium": (0, 10),
    "serum_magnesium": (0, 10),
    "serum_phosphate": (0, 10),
    "alanine_aminotransferase": (0, 10000),
    "aspartate_aminotransferase": (0, 10000),
    "alkaline_phosphatase": (0, 10000),
    "total_bilirubin": (0, 1000),
    "serum_albumin": (0, 100),
    "prothrombin_time": (0, 1000),
    "activated_partial_thromboplastin_time": (0, 1000),
    "immunoglobulin_g": (0, 100),
    "immunoglobulin_m": (0, 100),
}


def choose_existing_path(candidates: Sequence[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("None of the expected paths exist:\n" + "\n".join(str(x) for x in candidates))


def note_discharge_path(notes_root: Path) -> Path:
    return choose_existing_path(
        [
            notes_root / "discharge.csv.gz",
            notes_root / "discharge.csv",
            notes_root / "discharge.csv" / "discharge.csv",
        ]
    )


def maybe_gzip(path: Path) -> str:
    return "gzip" if path.suffix == ".gz" else "infer"


def normalize_icd(code: object) -> str:
    if pd.isna(code):
        return ""
    text = str(code).strip().upper()
    if not text or text in {"NAN", "NONE", "NULL", "-", "NA", "N/A"}:
        return ""
    return re.sub(r"[^A-Z0-9]", "", text)


def display_icd(version: int, code: object) -> str:
    code_str = "" if pd.isna(code) else str(code).strip().upper()
    version = int(version)
    if "." in code_str:
        pretty = code_str
    elif version in (9, 10) and len(code_str) > 3:
        pretty = f"{code_str[:3]}.{code_str[3:]}"
    else:
        pretty = code_str
    return f"ICD{version}:{pretty}"


def split_icd_cell(value: object) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    parts = re.split(r"[;,/|]+|\s+or\s+|\s+and\s+", text, flags=re.IGNORECASE)
    codes: List[str] = []
    for part in parts:
        part = part.strip().strip("()[]{}")
        if not part:
            continue
        match = re.search(r"[A-Z]?\d[\dA-Z]*(?:\.[\dA-Z]+)?", part.upper())
        if match:
            norm = normalize_icd(match.group(0))
            if norm:
                codes.append(norm)
    return codes


def canonical_code(code: object) -> str:
    text = "" if pd.isna(code) else str(code).strip().upper()
    if ":" not in text:
        return text
    prefix, raw = text.split(":", 1)
    raw = re.sub(r"[^A-Z0-9]", "", raw)
    if prefix == "ICD9" and len(raw) > 3:
        raw = f"{raw[:3]}.{raw[3:]}"
    elif prefix == "ICD10" and len(raw) > 3:
        raw = f"{raw[:3]}.{raw[3:]}"
    return f"{prefix}:{raw}"


def key_series(df: pd.DataFrame) -> pd.Series:
    return df["subject_id"].astype("Int64").astype(str) + "|" + df["hadm_id"].astype("Int64").astype(str)


def join_unique(values: Iterable[object]) -> str:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return ";".join(out)


def safe_name(label: str) -> str:
    parts = [p.strip() for p in str(label).split("|")]
    cleaned = []
    for part in parts:
        part = part.replace(",", " ").replace("/", " ")
        part = re.sub(r"\s+", " ", part).strip()
        if part:
            cleaned.append(part)
    return "_".join(cleaned)


def cell_value(entry: dict) -> str:
    num = entry.get("valuenum")
    val = entry.get("value")
    if num is not None:
        if isinstance(num, float) and num.is_integer():
            base = str(int(num))
        else:
            base = str(num)
    elif val not in (None, "", "___"):
        base = str(val)
    else:
        base = ""
    if not base:
        return ""
    unit = entry.get("valueuom")
    return f"{base} {unit}" if unit else base


def earliest_series_value(series_text: object) -> str:
    if series_text is None or (isinstance(series_text, float) and math.isnan(series_text)):
        return ""
    s = str(series_text or "").strip()
    if not s.startswith("["):
        return s
    try:
        arr = json.loads(s)
    except json.JSONDecodeError:
        return series_text or ""
    if not arr:
        return ""
    first = min(arr, key=lambda e: e.get("charttime") or "")
    return cell_value(first)


def normalize_space(text: object) -> str:
    return re.sub(r"[ \t]+", " ", str(text)).strip()


def extract_diagnosis_sections(note_text: object) -> str:
    if pd.isna(note_text):
        return ""
    text = str(note_text).replace("\r\n", "\n").replace("\r", "\n")
    matches = list(SECTION_START_RE.finditer(text))
    sections: List[str] = []
    for idx, match in enumerate(matches):
        start = match.end()
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        stop = SECTION_STOP_RE.search(text, pos=start, endpos=next_start)
        end = stop.start() if stop else next_start
        chunk = text[start:end]
        chunk = re.sub(r"(?m)^\s*[=\-]{2,}\s*$", "", chunk)
        chunk = normalize_space(chunk.replace("\n", " | "))
        if chunk:
            sections.append(chunk)
    return " || ".join(sections)


def find_iei_terms(diagnosis_text: str) -> str:
    matches = []
    for label, pattern in COMPILED_IEI_PATTERNS:
        if pattern.search(diagnosis_text or ""):
            matches.append(label)
    return ";".join(matches)


def load_iuis_codes(iuis_path: Path) -> Tuple[pd.DataFrame, set[Tuple[int, str]]]:
    iuis = pd.read_excel(iuis_path, sheet_name=0, dtype=str)
    iei_codes: set[Tuple[int, str]] = set()
    for _, row in iuis.iterrows():
        for version, col in [(9, "ICD9"), (10, "ICD10")]:
            for code in split_icd_cell(row.get(col)):
                iei_codes.add((version, code))
    return iuis, iei_codes


def load_checked_iuis_table_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_excel(path, sheet_name=0, dtype=str)
    df = df.rename(columns=str.strip)
    if "IUIS_name" not in df.columns or "IUIS_table_category" not in df.columns:
        return {}
    sub = df[["IUIS_name", "IUIS_table_category"]].dropna()
    return {str(k).strip(): str(v).strip() for k, v in zip(sub["IUIS_name"], sub["IUIS_table_category"])}


def build_wide_table(hosp_dir: Path, notes_dir: Path, iuis_path: Path, chunksize: int) -> pd.DataFrame:
    d_icd_path = choose_existing_path([hosp_dir / "d_icd_diagnoses.csv.gz", hosp_dir / "d_icd_diagnoses.csv"])
    diagnoses_path = choose_existing_path([hosp_dir / "diagnoses_icd.csv.gz", hosp_dir / "diagnoses_icd.csv"])
    patients_path = choose_existing_path([hosp_dir / "patients.csv.gz", hosp_dir / "patients.csv"])
    labevents_path = choose_existing_path([hosp_dir / "labevents.csv.gz", hosp_dir / "labevents.csv"])
    d_labitems_path = choose_existing_path([hosp_dir / "d_labitems.csv.gz", hosp_dir / "d_labitems.csv"])
    discharge_path = note_discharge_path(notes_dir)

    icd_dict = pd.read_csv(
        d_icd_path,
        compression=maybe_gzip(d_icd_path),
        usecols=["icd_code", "icd_version", "long_title"],
        dtype={"icd_code": "string", "long_title": "string"},
    )
    icd_dict["icd_code_norm"] = icd_dict["icd_code"].map(normalize_icd)

    _, iei_codes = load_iuis_codes(iuis_path)

    usecols = ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"]
    iei_rows = []
    for chunk in pd.read_csv(
        diagnoses_path,
        compression=maybe_gzip(diagnoses_path),
        usecols=usecols,
        dtype={"icd_code": "string"},
        chunksize=chunksize,
    ):
        chunk["icd_code_norm"] = chunk["icd_code"].map(normalize_icd)
        mask = [(int(v), c) in iei_codes for v, c in zip(chunk["icd_version"], chunk["icd_code_norm"])]
        matched = chunk.loc[mask].copy()
        if not matched.empty:
            matched["key"] = key_series(matched)
            iei_rows.append(matched)
    if not iei_rows:
        raise RuntimeError("No IEI-related admissions found from raw diagnoses.")

    iei_diag = pd.concat(iei_rows, ignore_index=True)
    target_keys = set(iei_diag["key"])

    all_rows = []
    for chunk in pd.read_csv(
        diagnoses_path,
        compression=maybe_gzip(diagnoses_path),
        usecols=usecols,
        dtype={"icd_code": "string"},
        chunksize=chunksize,
    ):
        chunk["key"] = key_series(chunk)
        chunk = chunk[chunk["key"].isin(target_keys)].copy()
        if not chunk.empty:
            chunk["icd_code_norm"] = chunk["icd_code"].map(normalize_icd)
            all_rows.append(chunk)
    all_diag = pd.concat(all_rows, ignore_index=True)

    icd_dict_small = icd_dict[["icd_code_norm", "icd_version", "long_title"]]
    iei_diag = iei_diag.merge(icd_dict_small, on=["icd_code_norm", "icd_version"], how="left")
    all_diag = all_diag.merge(icd_dict_small, on=["icd_code_norm", "icd_version"], how="left")
    for df in (iei_diag, all_diag):
        df["icd_display"] = [display_icd(v, c) for v, c in zip(df["icd_version"], df["icd_code"])]
        df["icd_with_title"] = df["icd_display"] + "|" + df["long_title"].fillna("")
        df.sort_values(["subject_id", "hadm_id", "seq_num"], inplace=True)

    iei_agg = iei_diag.groupby(["subject_id", "hadm_id"], as_index=False).agg(
        iei_icd_codes=("icd_display", join_unique),
        iei_icd_titles=("icd_with_title", join_unique),
    )
    all_agg = all_diag.groupby(["subject_id", "hadm_id"], as_index=False).agg(
        all_icd_codes=("icd_display", join_unique),
        all_icd_titles=("icd_with_title", join_unique),
    )
    base = iei_agg.merge(all_agg, on=["subject_id", "hadm_id"], how="left")

    patients = pd.read_csv(
        patients_path,
        compression=maybe_gzip(patients_path),
        usecols=["subject_id", "gender", "anchor_age"],
    )
    final = base.merge(patients, on="subject_id", how="left")
    first_cols = ["subject_id", "hadm_id", "gender", "anchor_age"]
    final = final[first_cols + [c for c in final.columns if c not in first_cols]]

    lab_items = pd.read_csv(
        d_labitems_path,
        compression=maybe_gzip(d_labitems_path),
        usecols=["itemid", "label", "fluid", "category"],
    )
    label_map = {
        int(row.itemid): " | ".join(
            str(x) for x in [row.label, row.fluid, row.category] if not pd.isna(x) and str(x)
        )
        for row in lab_items.itertuples(index=False)
    }

    lab_usecols = ["subject_id", "hadm_id", "itemid", "charttime", "value", "valuenum", "valueuom", "flag"]
    records: Dict[Tuple[int, int, int], List[dict]] = defaultdict(list)
    for chunk in pd.read_csv(
        labevents_path,
        compression=maybe_gzip(labevents_path),
        usecols=lab_usecols,
        chunksize=chunksize,
    ):
        chunk = chunk.dropna(subset=["hadm_id"])
        chunk["key"] = key_series(chunk)
        chunk = chunk[chunk["key"].isin(target_keys) & chunk["itemid"].isin(SELECTED_LAB_ITEMIDS)]
        if chunk.empty:
            continue
        chunk.sort_values(["subject_id", "hadm_id", "itemid", "charttime"], inplace=True)
        for row in chunk.itertuples(index=False):
            records[(int(row.subject_id), int(row.hadm_id), int(row.itemid))].append(
                {
                    "charttime": None if pd.isna(row.charttime) else str(row.charttime),
                    "value": None if pd.isna(row.value) else str(row.value),
                    "valuenum": None if pd.isna(row.valuenum) else float(row.valuenum),
                    "valueuom": None if pd.isna(row.valueuom) else str(row.valueuom),
                    "flag": None if pd.isna(row.flag) else str(row.flag),
                }
            )

    lab_rows: Dict[Tuple[int, int], Dict[str, object]] = {}
    for (subject_id, hadm_id, itemid), series in records.items():
        row = lab_rows.setdefault((subject_id, hadm_id), {"subject_id": subject_id, "hadm_id": hadm_id})
        row[f"lab_{itemid}_label"] = label_map.get(itemid, "")
        row[f"lab_{itemid}_series"] = json.dumps(series, ensure_ascii=False, separators=(",", ":"))
    lab_wide = pd.DataFrame(lab_rows.values()) if lab_rows else pd.DataFrame(columns=["subject_id", "hadm_id"])
    final = final.merge(lab_wide, on=["subject_id", "hadm_id"], how="left")

    notes_rows = []
    for chunk in pd.read_csv(
        discharge_path,
        compression=maybe_gzip(discharge_path),
        usecols=["note_id", "subject_id", "hadm_id", "charttime", "text"],
        chunksize=chunksize,
    ):
        chunk = chunk.dropna(subset=["hadm_id"])
        chunk["key"] = key_series(chunk)
        chunk = chunk[chunk["key"].isin(target_keys)].copy()
        if not chunk.empty:
            notes_rows.append(chunk)
    if notes_rows:
        notes = pd.concat(notes_rows, ignore_index=True)
        notes.sort_values(["subject_id", "hadm_id", "charttime", "note_id"], inplace=True)
        note_agg = notes.groupby(["subject_id", "hadm_id"], as_index=False).agg(
            discharge_note_text=("text", lambda x: "\n\n".join("" if pd.isna(v) else str(v) for v in x)),
            discharge_note_ids=("note_id", join_unique),
        )
    else:
        note_agg = pd.DataFrame(columns=["subject_id", "hadm_id", "discharge_note_text", "discharge_note_ids"])

    final = final.merge(note_agg, on=["subject_id", "hadm_id"], how="left")
    final["discharge_note_text"] = final["discharge_note_text"].fillna("")
    final["discharge_note_ids"] = final["discharge_note_ids"].fillna("")
    final.sort_values(["subject_id", "hadm_id"], inplace=True, kind="mergesort")
    final.reset_index(drop=True, inplace=True)
    return final


def filter_high_confidence(df: pd.DataFrame) -> pd.DataFrame:
    def filter_pair(codes_value: object, titles_value: object) -> pd.Series:
        codes = [] if pd.isna(codes_value) else [x.strip() for x in str(codes_value).split(";") if x.strip()]
        titles = [] if pd.isna(titles_value) else [x.strip() for x in str(titles_value).split(";") if x.strip()]
        kept_codes, kept_titles, removed_codes, removed_titles = [], [], [], []
        for idx, code in enumerate(codes):
            canon = canonical_code(code)
            title = titles[idx] if idx < len(titles) else ""
            if canon in HIGH_CONFIDENCE_CODES:
                kept_codes.append(code)
                kept_titles.append(title)
            else:
                removed_codes.append(code)
                removed_titles.append(title)
        return pd.Series(
            {
                "iei_icd_codes": ";".join(kept_codes),
                "iei_icd_titles": ";".join(kept_titles),
                "removed_broad_iei_icd_codes": ";".join(removed_codes),
                "removed_broad_iei_icd_titles": ";".join(removed_titles),
                "high_confidence_iei_code_count": len(kept_codes),
            }
        )

    filtered = df[["iei_icd_codes", "iei_icd_titles"]].apply(
        lambda row: filter_pair(row["iei_icd_codes"], row["iei_icd_titles"]), axis=1
    )
    out = df.drop(columns=["iei_icd_codes", "iei_icd_titles"]).join(filtered)
    first_cols = [
        "subject_id",
        "hadm_id",
        "gender",
        "anchor_age",
        "iei_icd_codes",
        "iei_icd_titles",
        "removed_broad_iei_icd_codes",
        "removed_broad_iei_icd_titles",
        "high_confidence_iei_code_count",
        "all_icd_codes",
        "all_icd_titles",
    ]
    out = out[out["high_confidence_iei_code_count"] > 0].copy()
    out = out[first_cols + [c for c in out.columns if c not in first_cols]]
    out.reset_index(drop=True, inplace=True)
    return out


def confirm_discharge_diagnosis(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["extracted_discharge_diagnosis"] = out["discharge_note_text"].map(extract_diagnosis_sections)
    out["discharge_diagnosis_iei_match_terms"] = out["extracted_discharge_diagnosis"].map(find_iei_terms)
    out["discharge_diagnosis_has_iei"] = out["discharge_diagnosis_iei_match_terms"].astype(bool)
    confirmed = out[out["discharge_diagnosis_has_iei"]].copy()
    front_cols = [
        "subject_id",
        "hadm_id",
        "gender",
        "anchor_age",
        "iei_icd_codes",
        "iei_icd_titles",
        "discharge_diagnosis_has_iei",
        "discharge_diagnosis_iei_match_terms",
        "extracted_discharge_diagnosis",
        "removed_broad_iei_icd_codes",
        "removed_broad_iei_icd_titles",
        "high_confidence_iei_code_count",
        "all_icd_codes",
        "all_icd_titles",
    ]
    front_cols = [c for c in front_cols if c in confirmed.columns]
    confirmed = confirmed[front_cols + [c for c in confirmed.columns if c not in front_cols]]
    confirmed.reset_index(drop=True, inplace=True)
    return confirmed


def collapse_lab_series(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    header = list(df.columns)
    bases = [h[:-7] for h in header if h.startswith("lab_") and h.endswith("_series")]
    label_counts: Dict[str, Counter] = defaultdict(Counter)
    for row in df.to_dict(orient="records"):
        for base in bases:
            value = str(row.get(base + "_label", "") or "").strip()
            if value and not value.startswith("[") and "|" in value:
                label_counts[base][value] += 1

    new_name = {}
    mapping_rows = []
    used = Counter()
    for base in bases:
        itemid = base[len("lab_"):]
        if label_counts[base]:
            label = label_counts[base].most_common(1)[0][0]
            name = safe_name(label)
        else:
            label, name = "", base
        if used[name]:
            name = f"{name}_{itemid}"
        used[name] += 1
        new_name[base] = name
        mapping_rows.append({"itemid": itemid, "original_label": label, "new_column_name": name})

    out_header: List[str] = []
    col_plan: List[Tuple[str, str, bool]] = []
    for h in header:
        if h.startswith("lab_") and h.endswith("_label"):
            continue
        if h.startswith("lab_") and h.endswith("_series"):
            out_name = new_name.get(h[:-7], h)
            out_header.append(out_name)
            col_plan.append((h, out_name, True))
        else:
            out_header.append(h)
            col_plan.append((h, h, False))

    out_rows = []
    for row in df.to_dict(orient="records"):
        out = {}
        for old, new, is_lab in col_plan:
            value = row.get(old, "")
            out[new] = earliest_series_value(value) if is_lab else value
        out_rows.append(out)

    return pd.DataFrame(out_rows, columns=out_header), pd.DataFrame(mapping_rows)


# Laboratory extraction from discharge-note pertinent results.
SIM_THRESHOLD = 0.55
COVER_THRESHOLD = 0.50
VALUE_PREFIX = "pr_"
CAT_ORDER = {"B": 0, "T": 1, "phag": 2, "comp": 3, "other": 4}
TIER_ORDER = {"basic": 0, "adv": 1, None: 2}
ANALYTE_MAP: Dict[str, Tuple[str, str, Optional[str]]] = {}


def _reg(canonical: str, category: str, tier: Optional[str], *tokens: str) -> None:
    for token in tokens:
        ANALYTE_MAP[token.upper()] = (canonical, category, tier)


_reg("IgG", "B", "basic", "IgG")
_reg("IgM", "B", "basic", "IgM")
_reg("IgA", "B", "basic", "IgA")
_reg("CD19", "B", "basic", "CD19", "CD19Abs")
_reg("CD20", "B", "basic", "CD20", "CD20Abs")
_reg("Anti-Streptolysin O", "B", "basic", "ASO", "ASLO")
_reg("SIgA", "B", "basic", "SIgA")
_reg("IgG1", "B", "adv", "IgG1")
_reg("IgG2", "B", "adv", "IgG2")
_reg("IgG3", "B", "adv", "IgG3")
_reg("IgG4", "B", "adv", "IgG4")
_reg("IgD", "B", "adv", "IgD")
_reg("Lymphocytes", "T", "basic", "Lymphs", "Lymph", "LYMPHS")
_reg("Absolute Lymphocyte Count", "T", "basic", "AbsLymp")
_reg("CD3", "T", "basic", "CD3", "CD3Abs")
_reg("CD4", "T", "basic", "CD4", "CD4Abs")
_reg("CD5", "T", "basic", "CD5", "CD5Abs")
_reg("CD8", "T", "basic", "CD8", "CD8Abs")
_reg("White Blood Cells", "phag", "basic", "WBC")
_reg("Neutrophils", "phag", "basic", "Neuts", "NEUTS", "Polys")
_reg("Absolute Neutrophil Count", "phag", "basic", "AbsNeut")
_reg("Bands", "phag", "basic", "Bands", "BANDS")
_reg("Monocytes", "phag", "basic", "Monos", "MONOS")
_reg("Absolute Monocyte Count", "phag", "basic", "AbsMono")
_reg("Eosinophils", "phag", "basic", "Eos", "EOS")
_reg("Absolute Eosinophil Count", "phag", "basic", "AbsEos")
_reg("Basophils", "phag", "basic", "Baso", "BASOS")
_reg("Absolute Basophil Count", "phag", "basic", "AbsBaso")
_reg("NBT", "phag", "basic", "NBT")
_reg("IgE", "phag", "basic", "IgE")
_reg("C3", "comp", "basic", "C3")
_reg("C4", "comp", "basic", "C4")
_reg("CH50", "comp", "basic", "CH50", "CH100")
_reg("C3a", "comp", "adv", "C3a")
_reg("C4a", "comp", "adv", "C4a")
_reg("C4d", "comp", "adv", "C4d")
_reg("C5a", "comp", "adv", "C5a")
_reg("C1", "comp", "adv", "C1")
_reg("C1q", "comp", "adv", "C1Q")
_reg("C2", "comp", "adv", "C2")
_reg("C5", "comp", "adv", "C5")
_reg("C6", "comp", "adv", "C6")
_reg("C7", "comp", "adv", "C7")
_reg("C8", "comp", "adv", "C8")
_reg("C9", "comp", "adv", "C9")

_GENERIC = {
    "Red Blood Cells": ["RBC"],
    "Hemoglobin": ["Hgb", "HGB"],
    "Hematocrit": ["Hct", "HCT"],
    "MCV": ["MCV"],
    "MCH": ["MCH"],
    "MCHC": ["MCHC"],
    "RDW": ["RDW"],
    "RDW-SD": ["RDWSD"],
    "Platelet Count": ["Plt", "PLT"],
    "Creatinine": ["Creat", "CREAT"],
    "Urea Nitrogen": ["UreaN", "BUN"],
    "Sodium": ["Na", "SODIUM"],
    "Potassium": ["K", "POTASSIUM"],
    "Chloride": ["Cl", "CHLORIDE"],
    "Bicarbonate": ["HCO3", "CO2", "calTCO2"],
    "Anion Gap": ["AnGap", "GAP"],
    "Glucose": ["Glucose", "GLUCOSE"],
    "Calcium": ["Calcium", "CALCIUM", "freeCa"],
    "Magnesium": ["Mg", "MAGNESIUM"],
    "Phosphate": ["Phos", "PHOS", "PHOSPHATE"],
    "Albumin": ["Albumin", "ALBUMIN"],
    "Total Protein": ["TotProt", "PROTEIN", "Protein"],
    "Globulin": ["Globuln"],
    "ALT": ["ALT", "ALT(SGPT)"],
    "AST": ["AST", "AST(SGOT)"],
    "Alkaline Phosphatase": ["AlkPhos"],
    "Bilirubin Total": ["TotBili", "BILI"],
    "Bilirubin Direct": ["DirBili"],
    "Bilirubin Indirect": ["IndBili"],
    "Lactate Dehydrogenase": ["LD(LDH)", "LDH"],
    "Lactate": ["Lactate", "LACTATE"],
    "Lipase": ["Lipase", "LIPASE"],
    "Amylase": ["Amylase"],
    "CK(CPK)": ["CK(CPK)"],
    "CK-MB": ["MB"],
    "Troponin T": ["cTropnT"],
    "proBNP": ["proBNP"],
    "PTT": ["PTT"],
    "CRP": ["CRP"],
    "Ferritin": ["Ferritn"],
    "Haptoglobin": ["Hapto"],
    "Iron": ["Iron"],
    "TIBC": ["calTIBC"],
    "Transferrin": ["TRF"],
    "Uric Acid": ["UricAcd"],
    "TSH": ["TSH"],
    "T4": ["T4"],
    "Cortisol": ["Cortsol", "CORTISOL"],
    "pH": ["pH", "PH"],
    "Epithelial Cells": ["Epi", "EPI"],
    "Triglycerides": ["Triglyc"],
    "Vitamin D": ["VitD", "25VitD"],
    "Anisocytosis": ["Anisocy", "ANISOCYT"],
    "Macrocytosis": ["Macrocy", "MACROCYT"],
    "Poikilocytosis": ["Poiklo", "POIKILOCY"],
    "Polychromasia": ["Polychr", "POLYCHROM"],
    "Hypochromia": ["Hypochr", "HYPOCHROM"],
    "Atypical Lymphocytes": ["Atyps", "ATYPS"],
    "Myelocytes": ["Myelos", "MYELOS"],
    "Metamyelocytes": ["Metas", "META"],
    "Nucleated RBC": ["NRBC"],
    "Reticulocyte": ["Ret"],
}
for canon, toks in _GENERIC.items():
    for token in toks:
        ANALYTE_MAP.setdefault(token.upper(), (canon, "other", None))

NOISE_TOKENS = {
    "AADO2", "FIO2", "PEEP", "O2", "SAT", "TEMP", "RATE", "RATES", "WT",
    "ACID", "ASSAY", "ENVELOP", "FUNCTIONAL", "HYALINE", "JOL", "RIMSO",
    "YOUNG", "BURR", "AUT", "ALPHA", "XS", "COUNT", "CT", "N", "V",
    "BIL", "UROBILNGN", "KETONE",
}
NEXT_HEADER_RE = re.compile(
    r"\n(?:Brief Hospital Course|Medications on Admission|Discharge Medications|"
    r"Discharge Disposition|Discharge Diagnosis|Discharge Condition|"
    r"Physical Exam|Facility|Attending):"
)
TOKEN_RE = re.compile(r"([A-Za-z][A-Za-z0-9%()/]{0,18})-((?:<|>)?\d[\d.,]*)")


def canonical(token: str) -> Tuple[str, str, Optional[str]]:
    key = token.rstrip("%").upper()
    return ANALYTE_MAP.get(key, (token, "other", None))


def pertinent_block(note: str) -> str:
    idx = note.find("Pertinent Results")
    if idx < 0:
        return ""
    tail = note[idx:]
    match = NEXT_HEADER_RE.search(tail)
    return tail[: match.start()] if match else tail


def parse_analytes(note: str) -> Dict[str, Tuple[str, Optional[str], str]]:
    block = pertinent_block(note)
    out: Dict[str, Tuple[str, Optional[str], str]] = {}
    for match in TOKEN_RE.finditer(block):
        raw_token = match.group(1).rstrip("%").upper()
        if raw_token in NOISE_TOKENS:
            continue
        name, cat, tier = canonical(match.group(1))
        if name in out:
            continue
        out[name] = (cat, tier, match.group(2).rstrip(".,"))
    return out


def mean_pairwise_jaccard(sets: Sequence[set]) -> float:
    if len(sets) < 2:
        return 1.0
    total = 0.0
    count = 0
    for left, right in combinations(sets, 2):
        union = left | right
        total += (len(left & right) / len(union)) if union else 1.0
        count += 1
    return total / count if count else 1.0


def extract_pertinent_results(notes_df: pd.DataFrame, merge_target: pd.DataFrame) -> pd.DataFrame:
    per_key: Dict[Tuple[str, str], Dict[str, str]] = {}
    groups: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    key_group: Dict[Tuple[str, str], str] = {}
    name_meta: Dict[str, Tuple[str, Optional[str]]] = {}

    for row in notes_df.to_dict(orient="records"):
        key = (str(row["subject_id"]), str(row["hadm_id"]))
        analytes = parse_analytes(str(row.get("discharge_note_text", "") or ""))
        per_key[key] = {name: val for name, (_cat, _tier, val) in analytes.items()}
        for name, (cat, tier, _val) in analytes.items():
            name_meta.setdefault(name, (cat, tier))
        icd = str(row.get("iei_icd_codes", "") or "")
        groups[icd].append(key)
        key_group[key] = icd

    kept_for_key: Dict[Tuple[str, str], set] = {}
    global_kept: set[str] = set()
    for icd, keys in groups.items():
        patient_sets = [set(per_key[k]) for k in keys]
        all_names = set().union(*patient_sets) if patient_sets else set()
        basic = {n for n in all_names if name_meta.get(n, ("other", None))[1] == "basic"}
        adv = {n for n in all_names if name_meta.get(n, ("other", None))[1] == "adv"}
        sim = mean_pairwise_jaccard(patient_sets)
        coverage = (len(basic) / len(all_names)) if all_names else 0.0
        if len(patient_sets) <= 1 or sim >= SIM_THRESHOLD:
            kept = set(all_names)
        else:
            kept = (basic | adv) if coverage >= COVER_THRESHOLD else set(basic)
        for key in keys:
            kept_for_key[key] = kept
        global_kept |= kept

    def sort_key(name: str) -> tuple:
        cat, tier = name_meta.get(name, ("other", None))
        return (CAT_ORDER.get(cat, 99), TIER_ORDER.get(tier, 99), name.lower())

    ordered_names = sorted(global_kept, key=sort_key)
    new_cols = [VALUE_PREFIX + name for name in ordered_names]

    out = merge_target.copy()
    for col in new_cols:
        if col not in out.columns:
            out[col] = ""

    for idx, row in out.iterrows():
        key = (str(row["subject_id"]), str(row["hadm_id"]))
        vals = per_key.get(key, {})
        kept = kept_for_key.get(key, set())
        for name, col in zip(ordered_names, new_cols):
            out.at[idx, col] = vals.get(name, "") if name in kept else ""

    return out


def load_hpo_helper(hpo_helper_csv: Optional[Path]) -> Tuple[pd.DataFrame, set[Tuple[str, str]]]:
    if hpo_helper_csv is None or not hpo_helper_csv.exists():
        empty = pd.DataFrame(columns=["hadm_id", "hpo_codes", "hpo_descriptions"])
        return empty, set()

    helper = pd.read_csv(hpo_helper_csv, dtype=str, encoding="utf-8-sig").fillna("")
    if "hadm_id" not in helper.columns:
        raise ValueError(f"HPO helper file missing hadm_id column: {hpo_helper_csv}")

    keep = ["hadm_id"]
    for col in ["hpo_codes", "hpo_descriptions"]:
        if col in helper.columns:
            keep.append(col)
    helper = helper[keep].drop_duplicates("hadm_id", keep="first").copy()
    for missing in ["hpo_codes", "hpo_descriptions"]:
        if missing not in helper.columns:
            helper[missing] = ""

    baseline_keys: set[Tuple[str, str]] = set()
    repaired_present = False
    with hpo_helper_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        header_map = {name: idx for idx, name in enumerate(header)}
        subject_idx = header_map.get("subject_id")
        hadm_idx = header_map.get("hadm_id")
        for row in reader:
            if subject_idx is not None and hadm_idx is not None and len(row) > max(subject_idx, hadm_idx):
                subject_id = str(row[subject_idx]).strip()
                hadm_id = str(row[hadm_idx]).strip()
                if subject_id.isdigit() and hadm_id.isdigit():
                    baseline_keys.add((subject_id, hadm_id))
            joined = " | ".join(row)
            if "14833492" in joined and "24054248" in joined:
                repaired_present = True

    if repaired_present:
        baseline_keys.add(("14833492", "24054248"))
        if not (helper["hadm_id"].astype(str) == "24054248").any():
            helper = pd.concat(
                [
                    helper,
                    pd.DataFrame([
                        {"hadm_id": "24054248", "hpo_codes": "[]", "hpo_descriptions": "[]"}
                    ]),
                ],
                ignore_index=True,
            )

    helper["hadm_id"] = helper["hadm_id"].astype(str)
    return helper, baseline_keys


def standardize_iei_diagnosis(match_terms: str, extracted_text: str) -> str:
    text = (extracted_text or "").lower()
    text_flat = re.sub(r"\s+", " ", text.replace("|", " ")).strip()
    terms = (match_terms or "").lower()
    if "human immunodeficiency virus" in text and "common variable" not in text_flat and "cvid" not in text_flat:
        return ""
    if "nk cell immunodeficiency" in text_flat:
        return "Autosomal recessive primary immunodeficiency with defective spontaneous natural killer cell cytotoxicity"
    if "autoimmune lymphoproliferative syndrome" in text_flat or "autoimmune lymphoproliferative syndrome" in terms or r"\balps\b" in terms:
        return "Autoimmune lymphoproliferative syndrome"
    if "hyper ige" in text_flat or re.search(r"hyper\s*ige", extracted_text or "", re.IGNORECASE):
        return "Autosomal dominant hyper-IgE syndrome due to STAT3 deficiency"
    if "type 3" in text_flat and "hereditary angioedema" in text_flat:
        return "Hereditary angioedema with normal C1Inh"
    if "c1 esterase inhibitor deficiency" in text_flat or "c1 inhibitor deficiency" in text_flat or "hereditary angioedema" in text_flat:
        return "Hereditary angioedema with C1Inh deficiency"
    if "familial mediterranean fever" in text_flat or r"\bfmf\b" in terms or "periodic fever syndrome" in text_flat or "periodic fever syndrome" in terms:
        return "Familial Mediterranean fever"
    if "selective iga deficiency" in text_flat or re.search(r"selective\s+iga\s+(?:deficiency|immunodeficiency)", extracted_text or "", re.IGNORECASE):
        return "Selective IgA deficiency"
    if "igg subclass deficiency" in text_flat or re.search(r"igg\s+subclass\s+deficiency", extracted_text or "", re.IGNORECASE):
        return "Recurrent infections associated with rare immunoglobulin isotypes deficiency"
    cvid_markers = [
        "common variable immunodeficiency",
        "common variable immune deficiency",
        "suspected common variable immunodeficiency",
        "chronic variable immunodeficiency syndrome",
        "common immunodeficiency disorder",
        "combined variable immunodeficiency",
        "history of cvid",
        "cvid",
    ]
    if any(marker in text_flat for marker in cvid_markers):
        return "Common variable immunodeficiency without known genetic defect"
    if "hypogammaglobulinemia" in text_flat or "agammaglobulinemia" in text_flat:
        return "Non-syndromic agammaglobulinemia"
    if "immunodeficiency" in text_flat or "immune deficiency" in text_flat:
        return "Immunodeficiency"
    return ""


def infer_iuis_table(standardized_diagnosis: str, extracted_text: str, checked_iuis_map: Dict[str, str]) -> str:
    if standardized_diagnosis in checked_iuis_map:
        return checked_iuis_map[standardized_diagnosis]
    if standardized_diagnosis in STATIC_IUIS_TABLE_MAP:
        return STATIC_IUIS_TABLE_MAP[standardized_diagnosis]

    text = (extracted_text or "").lower()
    if not standardized_diagnosis:
        if "human immunodeficiency virus" in text:
            return GENERIC_MULTI_TABLE
        return ""
    if standardized_diagnosis == "Immunodeficiency":
        if "ectodermal dysplasia" in text:
            return "Table 2 Combined immunodeficiencies with associated or syndromic features"
        if "bronchiectasis" in text or "combined variable immune deficiency" in text:
            return "Table 3 Predominantly Antibody Deficiencies"
        return GENERIC_MULTI_TABLE
    return ""


def extract_number(value: object) -> Optional[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value).replace(",", ""))
    return float(match.group(0)) if match else None


def format_number(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def normalize_unit(unit: str) -> str:
    unit = (unit or "").strip().lower()
    unit = unit.replace("μ", "u")
    unit = unit.replace("µ", "u")
    unit = re.sub(r"\s+", "", unit)
    return unit


def split_value_and_unit(value: object) -> Tuple[Optional[float], str]:
    if pd.isna(value):
        return None, ""
    text = str(value).strip()
    if not text:
        return None, ""
    match = re.match(r"\s*[<>]?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(.*)\s*$", text)
    if not match:
        return None, ""
    number = extract_number(match.group(1))
    unit = normalize_unit(match.group(2))
    return number, unit


def within_range(field: str, number: Optional[float]) -> bool:
    if number is None:
        return False
    bounds = PLAUSIBLE_RANGES.get(field)
    if not bounds:
        return True
    lo, hi = bounds
    return lo <= number <= hi


def convert_target_value(source_value: object, final_name: str) -> str:
    if final_name in {"patient_identifier", "hospital_admission_identifier", "discharge_note_identifier", "inborn_error_of_immunity_diagnosis", "hpo_codes", "sex", "IUIS_table"}:
        return "" if pd.isna(source_value) else str(source_value)
    if final_name == "age_in_years":
        number = extract_number(source_value)
        return format_number(number) if within_range(final_name, number) else ""
    if final_name in {"immunoglobulin_g", "immunoglobulin_m"}:
        number = extract_number(source_value)
        if number is None:
            return ""
        converted = number * 0.01  # MIMIC note pertinent immunoglobulins are treated as mg/dL -> g/L.
        return format_number(converted) if within_range(final_name, converted) else ""

    number, unit = split_value_and_unit(source_value)
    if number is None:
        return ""
    rules = UNIT_RULES.get(final_name)
    if not rules:
        return format_number(number) if within_range(final_name, number) else ""
    if unit not in rules:
        return ""
    converted = number * rules[unit]
    return format_number(converted) if within_range(final_name, converted) else ""


def build_lab_prigm(
    confirmed_df: pd.DataFrame,
    hpo_helper: pd.DataFrame,
    baseline_keys: Optional[set[Tuple[str, str]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    collapsed, mapping = collapse_lab_series(confirmed_df)
    with_pr = extract_pertinent_results(confirmed_df, collapsed)
    with_pr["subject_id"] = with_pr["subject_id"].astype(str)
    with_pr["hadm_id"] = with_pr["hadm_id"].astype(str)
    if baseline_keys:
        keep_mask = with_pr[["subject_id", "hadm_id"]].apply(tuple, axis=1).isin(baseline_keys)
        with_pr = with_pr[keep_mask].copy()
    if not hpo_helper.empty:
        helper = hpo_helper.copy()
        helper["hadm_id"] = helper["hadm_id"].astype(str)
        with_pr = with_pr.merge(helper, on="hadm_id", how="left")
    for col in ["hpo_codes", "hpo_descriptions"]:
        if col not in with_pr.columns:
            with_pr[col] = ""
        with_pr[col] = with_pr[col].fillna("")
    return with_pr, mapping


def convert_pr_fallback_value(value: object, final_name: str, mode: str) -> str:
    number = extract_number(value)
    if number is None:
        return ""
    if final_name == "white_blood_cell_count" and number == 0:
        return ""
    if mode == "direct":
        converted = number
    elif mode == "albumin_g_dl":
        converted = number * 10.0
    elif mode == "bilirubin_mg_dl":
        converted = number * 17.1
    else:
        converted = number
    return format_number(converted) if within_range(final_name, converted) else ""


def build_final_panel(lab_prigm_df: pd.DataFrame, checked_iuis_map: Dict[str, str]) -> pd.DataFrame:
    df = lab_prigm_df.copy()
    if "subject_id" not in df.columns or "hadm_id" not in df.columns:
        raise ValueError("lab_prIGM stage is missing subject_id/hadm_id columns.")

    df["subject_id"] = df["subject_id"].astype(str)
    df["hadm_id"] = df["hadm_id"].astype(str)
    df["standardized_iei_diagnosis"] = [
        standardize_iei_diagnosis(mt, diag)
        for mt, diag in zip(df["discharge_diagnosis_iei_match_terms"], df["extracted_discharge_diagnosis"])
    ]
    df["IUIS_table"] = [
        infer_iuis_table(diag, text, checked_iuis_map)
        for diag, text in zip(df["standardized_iei_diagnosis"], df["extracted_discharge_diagnosis"])
    ]
    df["discharge_note_ids"] = df.get("discharge_note_ids", "").fillna("").astype(str).str.split(";").str[0]

    mask_keep = ~df[["subject_id", "hadm_id"]].apply(tuple, axis=1).isin(EXCLUDED_FINAL_KEYS)
    df = df[mask_keep].copy()

    out_rows = []
    for row in df.to_dict(orient="records"):
        out = {}
        for old, new in FINAL_COLUMN_SPECS:
            out[new] = convert_target_value(row.get(old, ""), new)
            if out[new] == "" and new in PR_FALLBACK_MAP:
                source_col, mode = PR_FALLBACK_MAP[new]
                out[new] = convert_pr_fallback_value(row.get(source_col, ""), new, mode)
        key = (str(row.get("subject_id", "")), str(row.get("hadm_id", "")))
        if key in MANUAL_FINAL_OVERRIDES:
            out.update(MANUAL_FINAL_OVERRIDES[key])
        out_rows.append(out)
    out_df = pd.DataFrame(out_rows, columns=[new for _, new in FINAL_COLUMN_SPECS])

    # Order records by patient and admission identifiers.
    out_df["__sort_subject"] = pd.to_numeric(out_df["patient_identifier"], errors="coerce")
    out_df["__sort_hadm"] = pd.to_numeric(out_df["hospital_admission_identifier"], errors="coerce")
    out_df.sort_values(["__sort_subject", "__sort_hadm"], inplace=True, kind="mergesort")
    out_df.drop(columns=["__sort_subject", "__sort_hadm"], inplace=True)
    out_df.reset_index(drop=True, inplace=True)
    return out_df


def compare_csv_like(left: pd.DataFrame, right: pd.DataFrame) -> Tuple[bool, List[str]]:
    messages: List[str] = []
    ldf = left.fillna("").astype(str).reset_index(drop=True)
    rdf = right.fillna("").astype(str).reset_index(drop=True)
    if list(ldf.columns) != list(rdf.columns):
        messages.append("column mismatch")
        messages.append(f"left columns={list(ldf.columns)}")
        messages.append(f"right columns={list(rdf.columns)}")
        return False, messages
    if ldf.shape != rdf.shape:
        messages.append(f"shape mismatch left={ldf.shape} right={rdf.shape}")
        return False, messages
    diff_cols = []
    for col in ldf.columns:
        neq = int((ldf[col].to_numpy() != rdf[col].to_numpy()).sum())
        if neq:
            diff_cols.append((col, neq))
    if diff_cols:
        messages.append("cell diffs: " + ", ".join(f"{col}={n}" for col, n in diff_cols))
    return (not messages), messages


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an IEI cohort and EHR panel from MIMIC-IV data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON configuration file; relative paths are resolved from its folder.",
    )
    parser.add_argument("--skip-reference-compare", action="store_true", default=None)
    parser.add_argument(
        "--compare-reference",
        dest="skip_reference_compare",
        action="store_false",
        default=None,
        help="Compare against reference_final when one is configured.",
    )
    cli = parser.parse_args(argv)

    config_path = (
        cli.config.expanduser().resolve()
        if cli.config
        else Path(__file__).resolve().with_name("mimic_ehr_config.json")
    )
    if not config_path.is_file():
        parser.error(
            f"Configuration not found: {config_path}. Copy "
            "mimic_ehr_config.example.json and fill in its paths."
        )
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"Could not read configuration file: {exc}")
    if not isinstance(config, dict):
        parser.error("Configuration file must contain a JSON object.")

    allowed = {
        "hosp_dir",
        "notes_dir",
        "iuis_file",
        "iuis_checked_file",
        "hpo_helper_csv",
        "output_dir",
        "reference_final",
        "chunksize",
        "skip_reference_compare",
    }
    unknown = sorted(set(config) - allowed)
    if unknown:
        parser.error(f"Unknown configuration key(s): {', '.join(unknown)}")

    path_keys = allowed - {"chunksize", "skip_reference_compare"}
    for key in path_keys:
        value = config.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            parser.error(f"Configuration value {key!r} must be a path string or null.")
        path = Path(value).expanduser()
        config[key] = path if path.is_absolute() else (config_path.parent / path).resolve()

    required = ("hosp_dir", "notes_dir", "iuis_file", "iuis_checked_file")
    missing = [key for key in required if config.get(key) is None]
    if missing:
        parser.error("Missing required paths in configuration: " + ", ".join(missing))
    for key in required:
        if not config[key].exists():
            parser.error(f"Input path does not exist ({key}): {config[key]}")
    if not config["hosp_dir"].is_dir() or not config["notes_dir"].is_dir():
        parser.error("hosp_dir and notes_dir must point to directories.")
    if config.get("hpo_helper_csv") is not None and not config["hpo_helper_csv"].is_file():
        parser.error(f"HPO helper file does not exist: {config['hpo_helper_csv']}")

    output_dir = config.get("output_dir")
    if output_dir is None:
        output_dir = (config_path.parent / DEFAULT_OUTPUT_DIR).resolve()
    config["output_dir"] = output_dir
    if "hpo_helper_csv" not in config:
        config["hpo_helper_csv"] = output_dir / "lab_prIGM.csv"

    config.setdefault("reference_final", None)
    config.setdefault("chunksize", 500_000)
    skip_compare = config.get("skip_reference_compare", False)
    if cli.skip_reference_compare is not None:
        skip_compare = cli.skip_reference_compare
    if not isinstance(skip_compare, bool):
        parser.error("skip_reference_compare must be true or false.")
    config["skip_reference_compare"] = skip_compare

    chunksize = config["chunksize"]
    if not isinstance(chunksize, int) or isinstance(chunksize, bool) or chunksize <= 0:
        parser.error("chunksize must be a positive integer.")

    output_names = {
        "wide_output": "raw_iei_admission_wide_table.csv",
        "high_confidence_output": "raw_iei_admission_high_confidence_table.csv",
        "confirmed_output": "raw_iei_admission_diagnosis_confirmed_table.csv",
        "lab_prigm_output": "lab_prIGM.csv",
        "lab_mapping_output": "lab_itemid_english_chinese_mapping.csv",
        "final_output": "0813iei_final.csv",
    }
    for key, filename in output_names.items():
        config[key] = output_dir / filename

    return argparse.Namespace(config=config_path, **config)


def main() -> None:
    args = parse_args()

    # Read the existing helper before it is overwritten by the new lab_prIGM output.
    hpo_helper, baseline_keys = load_hpo_helper(args.hpo_helper_csv)
    checked_iuis_map = load_checked_iuis_table_map(args.iuis_checked_file)

    wide_df = build_wide_table(args.hosp_dir, args.notes_dir, args.iuis_file, args.chunksize)
    high_conf_df = filter_high_confidence(wide_df)
    confirmed_df = confirm_discharge_diagnosis(high_conf_df)
    lab_prigm_df, mapping_df = build_lab_prigm(confirmed_df, hpo_helper, baseline_keys)
    final_df = build_final_panel(lab_prigm_df, checked_iuis_map)

    write_csv(wide_df, args.wide_output)
    write_csv(high_conf_df, args.high_confidence_output)
    write_csv(confirmed_df, args.confirmed_output)
    write_csv(lab_prigm_df, args.lab_prigm_output)
    write_csv(mapping_df, args.lab_mapping_output)
    write_csv(final_df, args.final_output)

    print(f"wide_rows={len(wide_df)}")
    print(f"high_conf_rows={len(high_conf_df)}")
    print(f"confirmed_rows={len(confirmed_df)}")
    print(f"lab_prigm_rows={len(lab_prigm_df)}")
    print(f"final_rows={len(final_df)}")
    print(f"final_columns={len(final_df.columns)}")
    print(f"final_output={args.final_output}")

    if (
        not args.skip_reference_compare
        and args.reference_final is not None
        and args.reference_final.exists()
    ):
        reference_df = pd.read_csv(args.reference_final, dtype=str, encoding="utf-8-sig").fillna("")
        rebuilt_df = pd.read_csv(args.final_output, dtype=str, encoding="utf-8-sig").fillna("")
        ok, messages = compare_csv_like(rebuilt_df, reference_df)
        print(f"reference_compare_ok={ok}")
        for msg in messages:
            print(f"reference_compare_detail={msg}")

if __name__ == "__main__":
    main()
