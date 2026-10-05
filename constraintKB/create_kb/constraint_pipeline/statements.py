from __future__ import annotations

import copy
from typing import Any


_STATE_PREDICATES = {
    "present": "be present",
    "absent": "be absent",
    "positive": "test positive",
    "negative": "test negative",
    "low": "be low",
    "high": "be high",
    "normal": "be normal",
    "abnormal": "be abnormal",
    "increased": "be increased",
    "decreased": "be decreased",
    "variable": "be variable",
}

_NUMERIC_PREDICATES = {
    "equal": "equal",
    "less_than": "be less than",
    "less_or_equal": "be at most",
    "greater_than": "be greater than",
    "greater_or_equal": "be at least",
    "range": "be in the range",
}

_WEAK_STATE_PREDICATES = {
    "present": "may be present",
    "absent": "may be absent",
    "positive": "may test positive",
    "negative": "may test negative",
    "low": "may be low",
    "high": "may be high",
    "abnormal": "may be abnormal",
    "increased": "may be increased",
    "decreased": "may be decreased",
    "variable": "may be variable",
}

_WEAK_NUMERIC_PREDICATES = {
    "equal": "may equal",
    "less_than": "may be less than",
    "less_or_equal": "may be at most",
    "greater_than": "may be greater than",
    "greater_or_equal": "may be at least",
    "range": "may be in the range",
}


def _text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip(" .;")


def _measurement(value: Any, unit: Any) -> str:
    value_text = _text(value)
    unit_text = _text(unit)
    if not value_text:
        return unit_text
    if not unit_text:
        return value_text
    separator = "" if unit_text == "%" else " "
    return f"{value_text}{separator}{unit_text}"


def _short_qualifier(value: str) -> bool:
    return bool(value) and len(value.split()) <= 3 and not any(
        character in value for character in ",;:()"
    )


def render_constraint_statement(constraint: dict[str, Any]) -> str:
    """Render one standalone, deterministic disease-constraint assertion."""

    disease = constraint.get("disease")
    target = constraint.get("target")
    assertion = constraint.get("assertion")
    disease = disease if isinstance(disease, dict) else {}
    target = target if isinstance(target, dict) else {}
    assertion = assertion if isinstance(assertion, dict) else {}

    disease_name = (
        _text(disease.get("name") or disease.get("surface")) or "The disease"
    )
    target_name = _text(target.get("name") or target.get("surface")) or "the target"
    population = _text(constraint.get("population"))
    constraint_strength = constraint.get("constraint_strength", "strong")
    if constraint_strength == "weak":
        subject = (
            f"In {population} with {disease_name}, {target_name}"
            if population
            else f"In {disease_name}, {target_name}"
        )
    else:
        subject = (
            f"For {population}, {disease_name} requires"
            if population
            else f"{disease_name} requires"
        )

    operator = _text(assertion.get("operator"))
    amount = _measurement(assertion.get("value"), assertion.get("unit"))
    qualifier = _text(assertion.get("qualifier"))
    qualifier_consumed = False
    target_is_exclusion = target_name.casefold().startswith(
        ("exclusion of ", "absence of ")
    )
    negated = assertion.get("polarity") == "negated"
    if (
        constraint_strength == "strong"
        and operator == "absent"
        and target_is_exclusion
        and not negated
    ):
        statement = f"{subject} {target_name}"
    elif constraint_strength == "weak":
        if operator in _WEAK_NUMERIC_PREDICATES:
            predicate = _WEAK_NUMERIC_PREDICATES[operator]
            if amount:
                predicate = f"{predicate} {amount}"
        elif operator in _WEAK_STATE_PREDICATES:
            predicate = _WEAK_STATE_PREDICATES[operator]
        else:
            predicate = "may satisfy the stated condition"
        statement = f"{subject} {predicate}"
    else:
        if operator == "present" and _short_qualifier(qualifier):
            predicate = f"be {qualifier.lower()}"
            qualifier_consumed = True
        elif operator in _NUMERIC_PREDICATES:
            predicate = _NUMERIC_PREDICATES[operator]
            if amount:
                predicate = f"{predicate} {amount}"
        elif operator in _STATE_PREDICATES:
            predicate = _STATE_PREDICATES[operator]
        else:
            predicate = "satisfy the stated condition"
        infinitive = "not to" if negated else "to"
        statement = f"{subject} {target_name} {infinitive} {predicate}"

    context: list[str] = []
    temporality = _text(assertion.get("temporality"))
    frequency = _text(assertion.get("frequency"))
    certainty = _text(assertion.get("certainty"))
    if qualifier and not qualifier_consumed:
        context.append(qualifier)
    if temporality:
        context.append(f"temporality: {temporality}")
    if frequency:
        context.append(f"frequency: {frequency}")
    if constraint_strength == "weak" and certainty and certainty != "asserted":
        context.append(f"certainty: {certainty}")
    if context:
        statement = f"{statement} ({'; '.join(context)})"
    return f"{statement}."


def add_constraint_statements(
    constraints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return copies enriched with canonical, structure-derived statements."""

    enriched: list[dict[str, Any]] = []
    for constraint in constraints:
        value = copy.deepcopy(constraint)
        value["constraint_statement"] = render_constraint_statement(value)
        enriched.append(value)
    return enriched
