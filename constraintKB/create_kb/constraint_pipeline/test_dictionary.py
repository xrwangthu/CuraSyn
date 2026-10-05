from __future__ import annotations

import csv
import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


EHR_PUBLIC_PATH: Path | None = None
IEI_DICTIONARY_PATH: Path | None = None
HPO_OBO_PATH: Path | None = None
HPO_VOCABULARY_PATH: Path | None = None


def configure_scope(
    ehr_path: str,
    dictionary_path: str,
    hpo_path: str,
    hpo_vocabulary_path: str,
) -> None:
    """Configure one extraction run before starting its worker threads."""
    global EHR_PUBLIC_PATH, IEI_DICTIONARY_PATH, HPO_OBO_PATH, HPO_VOCABULARY_PATH
    values = {
        "ehr_path": ehr_path,
        "dictionary_path": dictionary_path,
        "hpo_path": hpo_path,
        "hpo_vocabulary_path": hpo_vocabulary_path,
    }
    for name, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Fill in extraction input path: {name}")
        if not Path(value).is_file():
            raise FileNotFoundError(f"{name} does not exist: {value}")
    EHR_PUBLIC_PATH, IEI_DICTIONARY_PATH, HPO_OBO_PATH, HPO_VOCABULARY_PATH = (
        Path(ehr_path).resolve(), Path(dictionary_path).resolve(), Path(hpo_path).resolve(),
        Path(hpo_vocabulary_path).resolve(),
    )
    for loader in (_load_dictionary_tests, load_allowed_tests, _load_hpo_ontology, load_allowed_hpo_terms):
        loader.cache_clear()
    load_allowed_tests()
    load_allowed_hpo_terms()


def _required_path(value: Path | None, configured: Path | None, name: str) -> Path:
    selected = value if value is not None else configured
    if selected is None:
        raise ValueError(f"Fill in extraction input path: {name}")
    return Path(selected).resolve()


EXAMINATION_PREFIX = "examination__"
PHENOTYPE_FIELD = "phenotype__hpo_codes"
NON_TEST_DICTIONARY_FIELDS = {
    "iei_icd_codes",
    "hpo_codes",
    "gender",
    "anchor_age",
}
REQUIRED_DICTIONARY_COLUMNS = {"cleaned_field", "meaning", "data_type", "unit"}

_REFERENCE_RANGE_PATTERNS = (
    re.compile(
        r"\bnormal\s+(?:reference\s+)?(?:range|ranges|interval|intervals|limits)\b",
        re.I,
    ),
    re.compile(r"\breference\s+(?:range|ranges|interval|intervals|values)\b", re.I),
    re.compile(r"\bwithin\s+(?:the\s+)?normal\s+(?:range|limits)\b", re.I),
    re.compile(r"\bnormal\s+values?\s+(?:are|for|of)\b", re.I),
    re.compile(r"\u6b63\u5e38(?:\u503c|\u53c2\u8003)?(?:\u8303\u56f4|\u533a\u95f4)"),
    re.compile(r"\u53c2\u8003(?:\u8303\u56f4|\u533a\u95f4)"),
)

_GENERIC_SURFACE_TOKENS = {
    "abnormal",
    "decreased",
    "elevated",
    "high",
    "increased",
    "level",
    "levels",
    "low",
    "plasma",
    "reduced",
    "result",
    "results",
    "serum",
    "test",
    "tests",
    "value",
    "values",
}

_SAFE_TEST_ALIASES = {
    "white_blood_cell_count": ("wbc", "wbc count"),
    "red_blood_cell_count": ("rbc", "rbc count"),
    "absolute_lymphocyte_count": ("alc",),
    "absolute_neutrophil_count": ("anc",),
    "immunoglobulin_g": ("igg",),
    "immunoglobulin_m": ("igm",),
    "hemoglobin_concentration": ("hemoglobin", "hgb", "hb"),
    "platelet_count": ("platelets", "platelet", "plt"),
    "alanine_aminotransferase": ("alt",),
    "aspartate_aminotransferase": ("ast",),

    "WBC": ("wbc count",),
    "RBC": ("rbc count",),
    "ALC": ("absolute lymphocyte count",),
    "ANC": ("absolute neutrophil count",),
    "INR_PT": ("inr", "international normalized ratio"),
    "pr_IgG": ("igg", "immunoglobulin g"),
    "pr_IgM": ("igm", "immunoglobulin m"),
    "pr_WBC": ("pertinent wbc", "pertinent wbc count"),
    "pr_Neutrophils": ("pertinent neutrophils",),
    "pr_Lymphocytes": ("pertinent lymphocytes",),
    "pr_ALC": ("pertinent alc",),
    "pr_ANC": ("pertinent anc",),
}


@dataclass(frozen=True, slots=True)
class AllowedTest:
    name: str
    meaning: str
    unit: str
    legacy_name: str


@dataclass(frozen=True, slots=True)
class AllowedHpoTerm:
    hpo_id: str
    name: str
    synonyms: tuple[str, ...]

    @property
    def target_name(self) -> str:
        return f"{self.hpo_id} - {self.name}"


def _snake_case(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


@lru_cache(maxsize=None)
def _load_dictionary_tests(
    path: Path | None = None,
) -> tuple[tuple[str, str, str], ...]:
    path = _required_path(path, IEI_DICTIONARY_PATH, "dictionary_path")
    if not path.is_file():
        raise FileNotFoundError(f"IEI test metadata does not exist: {path}")

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        if {"field", "meaning", "unit"} <= columns:
            rows = []
            seen = set()
            for row in reader:
                key, meaning, unit = (row[name].strip() for name in ("field", "meaning", "unit"))
                if not key or not meaning or not unit or key in seen:
                    raise ValueError("Dictionary field, meaning and unit must be nonempty; fields must be unique")
                seen.add(key)
                rows.append((key, meaning, unit))
            if not rows:
                raise ValueError("Dictionary contains no examination fields")
            return tuple(rows)
        missing = sorted(REQUIRED_DICTIONARY_COLUMNS - columns)
        if missing:
            raise ValueError(f"IEI test metadata is missing columns: {missing}")

        tests: list[tuple[str, str, str]] = []
        for row_number, row in enumerate(reader, 2):
            legacy_name = (row.get("cleaned_field") or "").strip()
            data_type = (row.get("data_type") or "").strip().casefold()
            if legacy_name in NON_TEST_DICTIONARY_FIELDS or data_type != "numeric":
                continue
            meaning = (row.get("meaning") or "").strip()
            unit = (row.get("unit") or "").strip()
            if not legacy_name or not meaning or not unit:
                raise ValueError(
                    f"IEI test metadata row {row_number} has an empty required value"
                )
            tests.append((legacy_name, meaning, unit))
    if not tests:
        raise ValueError(f"IEI metadata contains no numeric test fields: {path}")
    return tuple(tests)


@lru_cache(maxsize=None)
def load_allowed_tests(
    ehr_path: Path | None = None,
    dictionary_path: Path | None = None,
) -> tuple[AllowedTest, ...]:
    """Load the exact EHR examination columns and attach unit/alias metadata."""

    ehr_path = _required_path(ehr_path, EHR_PUBLIC_PATH, "ehr_path")
    if not ehr_path.is_file():
        raise FileNotFoundError(f"EHR scope file does not exist: {ehr_path}")
    with ehr_path.open(newline="", encoding="utf-8-sig") as handle:
        fieldnames = csv.DictReader(handle).fieldnames or []
    if PHENOTYPE_FIELD not in fieldnames and "hpo_codes" not in fieldnames:
        raise ValueError(f"EHR scope file is missing {PHENOTYPE_FIELD}: {ehr_path}")
    metadata = _load_dictionary_tests(dictionary_path)
    if "hpo_codes" in fieldnames:
        from src.ehr_contract import EXAMINATION_COLUMNS
        by_field = {row[0]: row for row in metadata}
        active = [name for name in EXAMINATION_COLUMNS if name in fieldnames]
        missing = [name for name in active if name not in by_field]
        if not active or missing:
            raise ValueError(f"Canonical EHR needs field/meaning/unit dictionary entries: {missing}")
        return tuple(AllowedTest(name=f"examination__{name}", meaning=by_field[name][1],
                                 unit=by_field[name][2], legacy_name=name) for name in active)
    examination_fields = [field for field in fieldnames if field.startswith(EXAMINATION_PREFIX)]
    if len(examination_fields) != len(metadata):
        raise ValueError("EHR examination fields and IEI unit metadata have different lengths")

    tests: list[AllowedTest] = []
    for field, (legacy_name, meaning, unit) in zip(
        examination_fields, metadata, strict=True
    ):
        expected_suffix = _snake_case(meaning)
        actual_suffix = field.removeprefix(EXAMINATION_PREFIX)
        if actual_suffix != expected_suffix:
            raise ValueError(
                "EHR examination field does not match IEI metadata ordering: "
                f"{field!r} != {meaning!r}"
            )
        tests.append(
            AllowedTest(
                name=field,
                meaning=meaning,
                unit=unit,
                legacy_name=legacy_name,
            )
        )
    return tuple(tests)


def allowed_test_names() -> frozenset[str]:
    return frozenset(test.name for test in load_allowed_tests())


def render_test_allowlist() -> str:
    return "\n".join(
        f"- {test.name}: {test.meaning} (unit metadata: {test.unit})"
        for test in load_allowed_tests()
    )


def _commit_obo_term(
    current: dict[str, Any], terms: dict[str, tuple[str, tuple[str, ...]]]
) -> None:
    hpo_id = current.get("id")
    name = current.get("name")
    if isinstance(hpo_id, str) and isinstance(name, str):
        terms[hpo_id] = (name, tuple(current.get("synonyms", ())))


@lru_cache(maxsize=None)
def _load_hpo_ontology(
    path: Path | None = None,
) -> dict[str, tuple[str, tuple[str, ...]]]:
    path = _required_path(path, HPO_OBO_PATH, "hpo_path")
    if not path.is_file():
        raise FileNotFoundError(f"HPO ontology does not exist: {path}")

    terms: dict[str, tuple[str, tuple[str, ...]]] = {}
    current: dict[str, Any] = {}
    synonym_pattern = re.compile(r'^synonym:\s+"((?:[^"\\]|\\.)*)"')
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "[Term]":
                _commit_obo_term(current, terms)
                current = {"synonyms": []}
            elif line.startswith("id: HP:"):
                current["id"] = line.removeprefix("id: ")
            elif line.startswith("name: "):
                current["name"] = line.removeprefix("name: ")
            else:
                match = synonym_pattern.match(line)
                if match:
                    current.setdefault("synonyms", []).append(
                        match.group(1).replace(r'\"', '"')
                    )
    _commit_obo_term(current, terms)
    if not terms:
        raise ValueError(f"HPO ontology contains no terms: {path}")
    return terms


@lru_cache(maxsize=None)
def load_allowed_hpo_terms(
    vocabulary_path: Path | None = None,
    obo_path: Path | None = None,
) -> tuple[AllowedHpoTerm, ...]:
    """Load the shared fixed task vocabulary and attach ontology names."""
    from src.schemas import load_hpo_vocabulary_file

    vocabulary_path = _required_path(
        vocabulary_path, HPO_VOCABULARY_PATH, "hpo_vocabulary_path",
    )
    ordered_codes, _ = load_hpo_vocabulary_file(vocabulary_path)

    ontology = _load_hpo_ontology(obo_path)
    missing = [code for code in ordered_codes if code not in ontology]
    if missing:
        raise ValueError(f"Task vocabulary HPO codes are missing from the ontology: {missing}")
    return tuple(
        AllowedHpoTerm(hpo_id=code, name=ontology[code][0], synonyms=ontology[code][1])
        for code in ordered_codes
    )


def allowed_hpo_target_names() -> frozenset[str]:
    return frozenset(term.target_name for term in load_allowed_hpo_terms())


def render_hpo_allowlist() -> str:
    return "\n".join(
        f"- {term.target_name}" for term in load_allowed_hpo_terms()
    )


def _surface_tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(re.findall(r"[a-z0-9]+", normalized))


def _contains_token_sequence(
    tokens: tuple[str, ...], sequence: tuple[str, ...]
) -> bool:
    if not sequence or len(sequence) > len(tokens):
        return False
    width = len(sequence)
    return any(
        tokens[index : index + width] == sequence
        for index in range(len(tokens) - width + 1)
    )


def test_target_surface_matches_ehr(name: Any, surface: Any) -> bool:
    if not isinstance(name, str) or not isinstance(surface, str):
        return False
    allowed = next((test for test in load_allowed_tests() if test.name == name), None)
    if allowed is None:
        return False
    surface_tokens = _surface_tokens(surface)
    if not surface_tokens:
        return False
    aliases = {
        _surface_tokens(allowed.legacy_name),
        _surface_tokens(allowed.meaning),
        *(
            _surface_tokens(alias)
            for alias in _SAFE_TEST_ALIASES.get(allowed.legacy_name, ())
        ),
    }
    for alias_tokens in aliases:
        if len(alias_tokens) > 1 and _contains_token_sequence(
            surface_tokens, alias_tokens
        ):
            return True
        if len(alias_tokens) == 1:
            substantive = tuple(
                token for token in surface_tokens if token not in _GENERIC_SURFACE_TOKENS
            )
            if substantive == alias_tokens:
                return True
    return False


def phenotype_target_surface_matches_hpo(name: Any, surface: Any) -> bool:
    if not isinstance(name, str) or not isinstance(surface, str):
        return False
    term = next(
        (term for term in load_allowed_hpo_terms() if term.target_name == name),
        None,
    )
    if term is None:
        return False
    surface_tokens = _surface_tokens(surface)
    aliases = [_surface_tokens(term.name)]
    aliases.extend(_surface_tokens(synonym) for synonym in term.synonyms)
    return any(
        _contains_token_sequence(surface_tokens, alias_tokens)
        for alias_tokens in aliases
        if alias_tokens
    )


def contains_reference_range_language(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return any(pattern.search(value) for pattern in _REFERENCE_RANGE_PATTERNS)


def test_constraint_exclusion_reasons(candidate: dict[str, Any]) -> list[str]:
    target = candidate.get("target")
    target = target if isinstance(target, dict) else {}
    if candidate.get("constraint_type") == "disease_phenotype":
        if target.get("name") not in allowed_hpo_target_names():
            return ["phenotype_target_not_in_task_vocabulary"]
        if not phenotype_target_surface_matches_hpo(
            target.get("name"), target.get("surface")
        ):
            return ["phenotype_target_surface_mismatch_hpo"]
        return []
    if candidate.get("constraint_type") != "disease_test_value":
        return []

    reasons: list[str] = []
    if target.get("name") not in allowed_test_names():
        reasons.append("test_target_not_in_ehr_public")
    elif not test_target_surface_matches_ehr(
        target.get("name"), target.get("surface")
    ):
        reasons.append("test_target_surface_mismatch_ehr_public")

    assertion = candidate.get("assertion")
    assertion = assertion if isinstance(assertion, dict) else {}
    if assertion.get("operator") == "normal":
        reasons.append("normal_test_state_excluded")

    claim_texts = [
        target.get("surface"),
        assertion.get("qualifier"),
        candidate.get("notes"),
    ]
    has_reference_range = any(
        contains_reference_range_language(value) for value in claim_texts
    )
    evidence = candidate.get("evidence")
    if assertion.get("operator") == "range" and isinstance(evidence, list):
        has_reference_range = has_reference_range or any(
            contains_reference_range_language(item.get("quote"))
            for item in evidence
            if isinstance(item, dict)
        )
    if has_reference_range:
        reasons.append("normal_reference_range_excluded")
    return reasons


def filter_constraints_to_scope(
    constraints: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for candidate in constraints:
        reasons = test_constraint_exclusion_reasons(candidate)
        if not reasons:
            kept.append(candidate)
            continue
        reason_counts.update(reasons)
        excluded.append({"exclusion_reasons": reasons, "constraint": candidate})
    return kept, excluded, {
        "input_constraints": len(constraints),
        "kept_constraints": len(kept),
        "excluded_constraints": len(excluded),
        "excluded_by_reason": dict(sorted(reason_counts.items())),
        "ehr_public_path": str(EHR_PUBLIC_PATH),
        "dictionary_path": str(IEI_DICTIONARY_PATH),
        "input_checksums": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (("ehr", EHR_PUBLIC_PATH), ("dictionary", IEI_DICTIONARY_PATH), ("hpo", HPO_OBO_PATH))
            if path is not None
        },
        "allowed_test_fields": len(allowed_test_names()),
        "hpo_ontology_path": str(HPO_OBO_PATH),
        "hpo_vocabulary_path": str(HPO_VOCABULARY_PATH),
        "allowed_hpo_terms": len(allowed_hpo_target_names()),
    }
