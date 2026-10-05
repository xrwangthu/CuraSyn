from __future__ import annotations

import copy
from typing import Any

from .test_dictionary import allowed_hpo_target_names, allowed_test_names

SCHEMA_VERSION = "3.1.0"
PROMPT_VERSION = "6.1.0"

CONSTRAINT_TYPES = {"disease_test_value", "disease_phenotype"}
CONSTRAINT_STRENGTHS = {"strong", "weak"}
TEST_TARGET_SUBTYPES = {"laboratory"}
PHENOTYPE_TARGET_SUBTYPES = {
    "sign",
    "symptom",
    "clinical_course",
    "onset",
    "trigger",
    "family_history",
    "comorbidity",
    "demographic",
    "other_phenotype",
}
OPERATORS = {
    "present",
    "absent",
    "positive",
    "negative",
    "low",
    "high",
    "abnormal",
    "increased",
    "decreased",
    "equal",
    "less_than",
    "less_or_equal",
    "greater_than",
    "greater_or_equal",
    "range",
    "variable",
    "other",
}
NUMERIC_OPERATORS = {
    "equal",
    "less_than",
    "less_or_equal",
    "greater_than",
    "greater_or_equal",
    "range",
}
TEST_STRONG_OPERATORS = NUMERIC_OPERATORS | {
    "positive",
    "negative",
    "abnormal",
}
TEST_WEAK_OPERATORS = TEST_STRONG_OPERATORS | {
    "low",
    "high",
    "increased",
    "decreased",
    "variable",
}
TEST_OPERATORS = TEST_STRONG_OPERATORS | TEST_WEAK_OPERATORS
PHENOTYPE_STRONG_OPERATORS = {"present", "absent"}
PHENOTYPE_WEAK_OPERATORS = PHENOTYPE_STRONG_OPERATORS | {"variable"}
PHENOTYPE_OPERATORS = PHENOTYPE_STRONG_OPERATORS | PHENOTYPE_WEAK_OPERATORS
STRONG_CERTAINTIES = {"asserted", "always"}
CERTAINTIES = {
    "asserted",
    "always",
    "usually",
    "often",
    "sometimes",
    "occasionally",
    "seldom",
    "rare",
    "possible",
    "variable",
    "uncertain",
}
WEAK_CERTAINTIES = CERTAINTIES - {"always"}
RELATION_ROLES = {
    "diagnostic_required",
    "diagnostic_supportive",
    "classification_candidate",
    "screening",
    "exclusionary",
    "differential",
    "characteristic",
    "epidemiologic",
    "prognostic",
    "other",
}
LOGIC_OPERATORS = {"SINGLE", "AND", "OR"}
EVIDENCE_ROLES = {"relation", "disease_context", "table_context"}
ISSUE_CATEGORIES = {
    "missing_constraint",
    "unsupported",
    "wrong_disease",
    "wrong_type",
    "wrong_assertion",
    "negation_or_qualifier",
    "non_atomic",
    "duplicate",
    "context_loss",
    "evidence_mismatch",
    "schema_error",
    "coverage_error",
    "other",
}


def _nullable_string() -> dict[str, Any]:
    return {"type": ["string", "null"]}


EVIDENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["block_id", "quote", "role"],
    "properties": {
        "block_id": {"type": "string"},
        "quote": {"type": "string"},
        "role": {"type": "string", "enum": sorted(EVIDENCE_ROLES)},
    },
}

CONSTRAINT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "local_id",
        "constraint_type",
        "constraint_strength",
        "disease",
        "target",
        "assertion",
        "relation_role",
        "population",
        "logic",
        "evidence",
        "notes",
    ],
    "properties": {
        "local_id": {"type": "string"},
        "constraint_type": {"type": "string", "enum": sorted(CONSTRAINT_TYPES)},
        "constraint_strength": {
            "type": "string",
            "enum": sorted(CONSTRAINT_STRENGTHS),
        },
        "disease": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "surface"],
            "properties": {
                "name": {"type": "string"},
                "surface": {"type": "string"},
            },
        },
        "target": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "surface", "subtype"],
            "properties": {
                "name": {"type": "string"},
                "surface": {"type": "string"},
                "subtype": {
                    "type": "string",
                    "enum": sorted(TEST_TARGET_SUBTYPES | PHENOTYPE_TARGET_SUBTYPES),
                },
            },
        },
        "assertion": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "polarity",
                "operator",
                "value",
                "unit",
                "frequency",
                "certainty",
                "qualifier",
                "temporality",
            ],
            "properties": {
                "polarity": {"type": "string", "enum": ["affirmed", "negated"]},
                "operator": {"type": "string", "enum": sorted(OPERATORS)},
                "value": _nullable_string(),
                "unit": _nullable_string(),
                "frequency": _nullable_string(),
                "certainty": {"type": "string", "enum": sorted(CERTAINTIES)},
                "qualifier": _nullable_string(),
                "temporality": _nullable_string(),
            },
        },
        "relation_role": {"type": "string", "enum": sorted(RELATION_ROLES)},
        "population": _nullable_string(),
        "logic": {
            "type": "object",
            "additionalProperties": False,
            "required": ["group_id", "operator", "group_text"],
            "properties": {
                "group_id": _nullable_string(),
                "operator": {"type": "string", "enum": sorted(LOGIC_OPERATORS)},
                "group_text": _nullable_string(),
            },
        },
        "evidence": {"type": "array", "minItems": 1, "items": EVIDENCE_SCHEMA},
        "notes": _nullable_string(),
    },
}

ACTOR_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["reviewed_primary_block_ids", "constraints"],
    "properties": {
        "reviewed_primary_block_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "constraints": {"type": "array", "items": CONSTRAINT_SCHEMA},
    },
}

CRITIC_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "verdict",
        "coverage_complete",
        "reviewed_primary_block_ids",
        "issues",
    ],
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "revise"]},
        "coverage_complete": {"type": "boolean"},
        "reviewed_primary_block_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "issue_id",
                    "category",
                    "candidate_id",
                    "block_id",
                    "evidence_quote",
                    "message",
                    "required_action",
                ],
                "properties": {
                    "issue_id": {"type": "string"},
                    "category": {"type": "string", "enum": sorted(ISSUE_CATEGORIES)},
                    "candidate_id": _nullable_string(),
                    "block_id": _nullable_string(),
                    "evidence_quote": _nullable_string(),
                    "message": {"type": "string"},
                    "required_action": {"type": "string"},
                },
            },
        },
    },
}


def json_schema_name(kind: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in kind)
    return f"constraint_pipeline_{safe}_{SCHEMA_VERSION.replace('.', '_')}"


def actor_result_schema(
    constraint_type: str,
    primary_block_refs: list[str],
    all_block_refs: list[str],
) -> dict[str, Any]:
    if constraint_type not in CONSTRAINT_TYPES:
        raise ValueError(f"Unknown constraint type: {constraint_type}")
    schema = copy.deepcopy(ACTOR_RESULT_SCHEMA)
    reviewed = schema["properties"]["reviewed_primary_block_ids"]
    reviewed["items"]["enum"] = list(primary_block_refs)
    reviewed["minItems"] = len(primary_block_refs)
    reviewed["maxItems"] = len(primary_block_refs)

    candidate = schema["properties"]["constraints"]["items"]
    candidate["properties"]["constraint_type"]["enum"] = [constraint_type]
    target_subtypes = (
        TEST_TARGET_SUBTYPES
        if constraint_type == "disease_test_value"
        else PHENOTYPE_TARGET_SUBTYPES
    )
    candidate["properties"]["target"]["properties"]["subtype"]["enum"] = sorted(
        target_subtypes
    )
    if constraint_type == "disease_test_value":
        candidate["properties"]["target"]["properties"]["name"]["enum"] = sorted(
            allowed_test_names()
        )
    else:
        candidate["properties"]["target"]["properties"]["name"]["enum"] = sorted(
            allowed_hpo_target_names()
        )
    assertion = candidate["properties"]["assertion"]["properties"]
    assertion["polarity"]["enum"] = ["affirmed"]
    assertion["operator"]["enum"] = sorted(
        TEST_OPERATORS
        if constraint_type == "disease_test_value"
        else PHENOTYPE_OPERATORS
    )
    assertion["certainty"]["enum"] = sorted(CERTAINTIES)
    candidate["properties"]["evidence"]["items"]["properties"]["block_id"][
        "enum"
    ] = list(all_block_refs)
    return schema


def critic_result_schema(
    primary_block_refs: list[str],
    all_block_refs: list[str],
) -> dict[str, Any]:
    schema = copy.deepcopy(CRITIC_RESULT_SCHEMA)
    reviewed = schema["properties"]["reviewed_primary_block_ids"]
    reviewed["items"]["enum"] = list(primary_block_refs)
    reviewed["minItems"] = len(primary_block_refs)
    reviewed["maxItems"] = len(primary_block_refs)
    block_id = schema["properties"]["issues"]["items"]["properties"]["block_id"]
    block_id["enum"] = [None, *all_block_refs]
    schema["properties"]["issues"]["items"]["properties"]["issue_id"][
        "minLength"
    ] = 1
    return schema
