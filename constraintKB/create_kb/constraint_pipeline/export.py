#!/usr/bin/env python3
"""Convert final constraint JSONL into the disease-grouped reference JSON format."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence


SUPPORTED_TYPES = {"disease_phenotype", "disease_test_value"}
COMPARISON_SYMBOLS = {
    "less_than": "<",
    "less_or_equal": "<=",
    "greater_than": ">",
    "greater_or_equal": ">=",
    "equal": "=",
}


class ConversionError(ValueError):
    """Raised when an input record violates the input/output contract."""


def _nonempty_string(value: Any, field: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConversionError(f"line {line_number}: {field} must be a non-empty string")
    return value.strip()


def read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ConversionError(
                    f"line {line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ConversionError(
                    f"line {line_number}: each JSONL record must be an object"
                )
            records.append((line_number, record))
    return records


def _strength_from_record(record: dict[str, Any], line_number: int) -> str:
    assertion = record.get("assertion")
    if isinstance(assertion, dict):
        frequency = assertion.get("frequency")
        if isinstance(frequency, str) and frequency.strip():
            return frequency.strip()
        certainty = assertion.get("certainty")
        if isinstance(certainty, str) and certainty.strip():
            return certainty.strip()

    return _nonempty_string(
        record.get("constraint_strength"),
        "assertion.frequency, assertion.certainty, or constraint_strength",
        line_number,
    )


def _plain_text(value: str) -> str:
    value = re.sub(
        r"<sup>\s*(.*?)\s*</sup>",
        lambda match: f"^{match.group(1)}",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"<[^>]+>", "", value)
    return " ".join(html.unescape(value).split())


def _render_test_constraint(record: dict[str, Any], line_number: int) -> str | None:
    target = record.get("target")
    assertion = record.get("assertion")
    if not isinstance(target, dict) or not isinstance(assertion, dict):
        return None

    surface_value = target.get("surface")
    operator_value = assertion.get("operator")
    if not isinstance(surface_value, str) or not surface_value.strip():
        return None
    if not isinstance(operator_value, str) or not operator_value.strip():
        return None

    surface = _plain_text(surface_value.strip())
    operator = operator_value.strip()
    value = assertion.get("value")
    unit = assertion.get("unit")
    rendered_value = _plain_text(str(value)) if value is not None else ""
    rendered_unit = _plain_text(unit) if isinstance(unit, str) and unit.strip() else ""
    normalized_surface = surface.casefold()

    # A source surface that already contains a comparator and value is the most faithful compact form.
    if re.search(r"(?:<=|>=|<|>|=|≤|≥)\s*[^\s]*\d", surface):
        return surface

    if operator in COMPARISON_SYMBOLS:
        if not rendered_value or re.search(r"\d", rendered_value) is None:
            return None
        parts = [surface, COMPARISON_SYMBOLS[operator], rendered_value]
        if rendered_unit:
            parts.append(rendered_unit)
        return " ".join(parts)

    if operator == "range":
        if not rendered_value or re.search(r"\d", rendered_value) is None:
            return None
        parts = [surface, rendered_value]
        if rendered_unit:
            parts.append(rendered_unit)
        return " ".join(parts)

    return None


def convert_records(
    records: Iterable[tuple[int, dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, int]]:
    groups: dict[str, dict[str, Any]] = {}
    phenotype_count = 0
    test_count = 0
    dropped_test_count = 0
    record_count = 0

    for line_number, record in records:
        record_count += 1
        constraint_type = _nonempty_string(
            record.get("constraint_type"), "constraint_type", line_number
        )
        if constraint_type not in SUPPORTED_TYPES:
            raise ConversionError(
                f"line {line_number}: unsupported constraint_type {constraint_type!r}; "
                f"expected one of {sorted(SUPPORTED_TYPES)}"
            )

        disease = record.get("disease")
        if not isinstance(disease, dict):
            raise ConversionError(f"line {line_number}: disease must be an object")
        disease_name = _nonempty_string(disease.get("name"), "disease.name", line_number)

        if constraint_type == "disease_test_value":
            compact_constraint = _render_test_constraint(record, line_number)
            if compact_constraint is None:
                dropped_test_count += 1
                continue
            output_field = "THEN"
            output_item = {
                "CONSTRAINT": compact_constraint,
                "STRENGTH": _strength_from_record(record, line_number),
            }
        else:
            target = record.get("target")
            if not isinstance(target, dict):
                raise ConversionError(f"line {line_number}: target must be an object")
            hpo_name = _nonempty_string(target.get("name"), "target.name", line_number)
            if not hpo_name.startswith("HP:"):
                raise ConversionError(
                    f"line {line_number}: phenotype target.name must begin with 'HP:'"
                )
            output_field = "HPO"
            output_item = {
                "CONSTRAINT": f"HPO:{hpo_name.removeprefix('HP:')}",
                "STRENGTH": _strength_from_record(record, line_number),
            }

        group = groups.setdefault(
            disease_name,
            {"IF": disease_name, "THEN": [], "HPO": []},
        )
        group[output_field].append(output_item)
        if output_field == "THEN":
            test_count += 1
        else:
            phenotype_count += 1

    output_groups: dict[str, dict[str, Any]] = {}
    for group_index, group in enumerate(groups.values(), start=1):
        output_group: dict[str, Any] = {"IF": group["IF"]}
        if group["THEN"]:
            output_group["THEN"] = group["THEN"]
        if group["HPO"]:
            output_group["HPO"] = group["HPO"]
        output_groups[str(group_index)] = output_group

    output = {"constraints": output_groups}
    stats = {
        "records": record_count,
        "kept": test_count + phenotype_count,
        "dropped_tests": dropped_test_count,
        "groups": len(output_groups),
        "then": test_count,
        "hpo": phenotype_count,
    }
    return output, stats


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def convert_file(input_path: Path, output_path: Path) -> dict[str, int]:
    if not input_path.is_file():
        raise ConversionError(f"input JSONL does not exist: {input_path}")
    payload, stats = convert_records(read_jsonl(input_path))
    write_json_atomic(output_path, payload)
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert final constraints.jsonl into grouped JSON with compact test values "
            "in THEN and phenotypes in HPO."
        )
    )
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        stats = convert_file(args.input_jsonl, args.output_json)
    except (ConversionError, OSError) as exc:
        raise SystemExit(f"conversion failed: {exc}") from exc
    print(
        "converted "
        f"{stats['kept']} of {stats['records']} records into {stats['groups']} disease groups "
        f"(THEN={stats['then']}, HPO={stats['hpo']}, "
        f"dropped_tests={stats['dropped_tests']}): {args.output_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
