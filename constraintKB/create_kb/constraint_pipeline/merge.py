from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from typing import Any

from .markdown import Block, Chunk, Document
from .schema import SCHEMA_VERSION
from .statements import render_constraint_statement


def _normalized(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^\w.%<>=+-]+", " ", text, flags=re.UNICODE).strip()


def _hash(value: Any, length: int = 20) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]


def semantic_key(constraint: dict[str, Any]) -> tuple[str, ...]:
    assertion = constraint["assertion"]
    logic = constraint["logic"]
    return (
        constraint["constraint_type"],
        constraint.get("constraint_strength", "strong"),
        _normalized(constraint["disease"]["name"]),
        _normalized(constraint["target"]["name"]),
        _normalized(constraint["target"]["subtype"]),
        _normalized(assertion["polarity"]),
        _normalized(assertion["operator"]),
        _normalized(assertion["value"]),
        _normalized(assertion["unit"]),
        _normalized(assertion["frequency"]),
        _normalized(assertion["certainty"]),
        _normalized(assertion["qualifier"]),
        _normalized(assertion["temporality"]),
        _normalized(constraint["relation_role"]),
        _normalized(constraint["population"]),
        _normalized(logic["operator"]),
        _normalized(logic["group_text"]),
    )


def conflict_key(constraint: dict[str, Any]) -> tuple[str, ...]:
    return (
        constraint["constraint_type"],
        constraint.get("constraint_strength", "strong"),
        _normalized(constraint["disease"]["name"]),
        _normalized(constraint["target"]["name"]),
        _normalized(constraint["population"]),
        _normalized(constraint["assertion"]["temporality"]),
    )


def constraint_identifier(constraint: dict[str, Any]) -> str:
    return f"K-{_hash(semantic_key(constraint))}"


def _evidence_key(evidence: dict[str, Any]) -> tuple[str, str, str]:
    quote_hash = evidence.get("quote_sha256")
    if not isinstance(quote_hash, str) or not quote_hash:
        quote = evidence.get("quote", "")
        quote_hash = hashlib.sha256(str(quote).encode("utf-8")).hexdigest()
    return (
        str(evidence.get("block_id", "")),
        quote_hash,
        str(evidence.get("role", "")),
    )


def _deduplicate_evidence(
    evidence_items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    duplicates = 0
    for evidence in evidence_items:
        key = _evidence_key(evidence)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(copy.deepcopy(evidence))
    return unique, duplicates


def _append_duplicate_provenance(
    target: dict[str, Any], duplicate: dict[str, Any]
) -> None:
    candidates = [duplicate.get("provenance"), *duplicate.get("duplicate_provenance", [])]
    candidates = [item for item in candidates if isinstance(item, dict)]
    if not candidates:
        return

    sources = target.setdefault("duplicate_provenance", [])
    known = {
        json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for item in [target.get("provenance"), *sources]
        if isinstance(item, dict)
    }
    for provenance in candidates:
        encoded = json.dumps(
            provenance, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if encoded in known:
            continue
        sources.append(copy.deepcopy(provenance))
        known.add(encoded)


def deduplicate_constraints(
    constraints: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_semantic: dict[tuple[str, ...], dict[str, Any]] = {}
    constraint_duplicates = 0
    evidence_duplicates = 0
    for constraint in constraints:
        key = semantic_key(constraint)
        if key not in by_semantic:
            value = copy.deepcopy(constraint)
            value["evidence"], removed = _deduplicate_evidence(value["evidence"])
            evidence_duplicates += removed
            by_semantic[key] = value
            continue

        constraint_duplicates += 1
        existing = by_semantic[key]
        existing["evidence"], removed = _deduplicate_evidence(
            [*existing["evidence"], *constraint["evidence"]]
        )
        evidence_duplicates += removed
        _append_duplicate_provenance(existing, constraint)

    unique = list(by_semantic.values())
    return unique, {
        "input_constraints": len(constraints),
        "semantic_duplicates_merged": constraint_duplicates,
        "duplicate_evidence_removed": evidence_duplicates,
        "deduplicated_constraints": len(unique),
    }


def _compact_group_key(constraint: dict[str, Any]) -> tuple[str, ...]:
    logic = constraint["logic"]
    if logic["operator"] in {"AND", "OR"} and _normalized(logic["group_text"]):
        return (
            "logical_group",
            constraint["constraint_type"],
            constraint.get("constraint_strength", "strong"),
            _normalized(constraint["disease"]["name"]),
            _normalized(constraint["population"]),
            _normalized(logic["operator"]),
            _normalized(logic["group_text"]),
        )
    return ("single_constraint", *semantic_key(constraint))


def compact_constraints(
    constraints: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Normalize repeated rule context while preserving every atomic constraint."""

    groups_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    evidence_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    evidence_references = 0

    for constraint in constraints:
        group_key = _compact_group_key(constraint)
        group = groups_by_key.get(group_key)
        if group is None:
            group_id = f"G-{_hash(group_key)}"
            group = {
                "group_id": group_id,
                "constraint_type": constraint["constraint_type"],
                "constraint_strength": constraint.get(
                    "constraint_strength", "strong"
                ),
                "disease": copy.deepcopy(constraint["disease"]),
                "population": constraint["population"],
                "logic": {
                    "operator": constraint["logic"]["operator"],
                    "group_text": constraint["logic"]["group_text"],
                },
                "source_group_ids": [],
                "_constraint_sets": {},
            }
            groups_by_key[group_key] = group

        source_group_id = constraint["logic"].get("group_id")
        if source_group_id and source_group_id not in group["source_group_ids"]:
            group["source_group_ids"].append(source_group_id)

        evidence_ids: list[str] = []
        for evidence in constraint["evidence"]:
            evidence_references += 1
            evidence_key = _evidence_key(evidence)
            evidence_record = evidence_by_key.get(evidence_key)
            if evidence_record is None:
                evidence_id = f"E-{_hash(evidence_key)}"
                evidence_record = {"evidence_id": evidence_id, **copy.deepcopy(evidence)}
                evidence_by_key[evidence_key] = evidence_record
            evidence_id = evidence_record["evidence_id"]
            if evidence_id not in evidence_ids:
                evidence_ids.append(evidence_id)

        set_key = (
            json.dumps(
                constraint["assertion"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            _normalized(constraint["relation_role"]),
            *evidence_ids,
        )
        constraint_sets = group["_constraint_sets"]
        constraint_set = constraint_sets.get(set_key)
        if constraint_set is None:
            constraint_set = {
                "constraint_set_id": f"S-{_hash((group['group_id'], set_key))}",
                "assertion": copy.deepcopy(constraint["assertion"]),
                "relation_role": constraint["relation_role"],
                "evidence_ids": evidence_ids,
                "targets": [],
            }
            constraint_sets[set_key] = constraint_set

        target = {
            "constraint_id": constraint.get("constraint_id")
            or constraint_identifier(constraint),
            "constraint_statement": render_constraint_statement(constraint),
            "target": copy.deepcopy(constraint["target"]),
            "notes": constraint["notes"],
        }
        if "local_id" in constraint:
            target["source_local_id"] = constraint["local_id"]
        if constraint["disease"].get("surface") != group["disease"].get("surface"):
            target["disease_surface"] = constraint["disease"].get("surface")
        for field in ("provenance", "duplicate_provenance"):
            if field in constraint:
                target[field] = copy.deepcopy(constraint[field])
        constraint_set["targets"].append(target)

    groups: list[dict[str, Any]] = []
    constraint_set_count = 0
    compact_evidence_references = 0
    for group in groups_by_key.values():
        constraint_sets = list(group.pop("_constraint_sets").values())
        group["constraint_sets"] = constraint_sets
        constraint_set_count += len(constraint_sets)
        compact_evidence_references += sum(
            len(constraint_set["evidence_ids"])
            for constraint_set in constraint_sets
        )
        groups.append(group)

    evidence_records = list(evidence_by_key.values())
    return groups, evidence_records, {
        "atomic_constraints": len(constraints),
        "constraint_groups": len(groups),
        "constraint_sets": constraint_set_count,
        "shared_assertion_targets": len(constraints) - constraint_set_count,
        "input_evidence_references": evidence_references,
        "compact_evidence_references": compact_evidence_references,
        "duplicate_constraint_evidence_references_elided": evidence_references
        - compact_evidence_references,
        "unique_evidence_records": len(evidence_records),
        "repeated_evidence_payloads_elided": evidence_references
        - len(evidence_records),
    }


def materialize_constraint(
    candidate: dict[str, Any],
    *,
    document: Document,
    chunk: Chunk,
    blocks: dict[str, Block],
    actor_round: int,
    actor_model: str,
    critic_model: str,
) -> dict[str, Any]:
    value = copy.deepcopy(candidate)
    actor_local_id = value.pop("local_id")
    logic = value["logic"]
    if logic["group_id"] is not None:
        logic["group_id"] = (
            f"{document.document_id}:{chunk.chunk_id.split(':')[-1]}:"
            f"{value['constraint_type']}:{logic['group_id']}"
        )
    expanded_evidence: list[dict[str, Any]] = []
    for evidence in value["evidence"]:
        block = blocks[evidence["block_id"]]
        char_start = block.text.index(evidence["quote"])
        expanded = dict(evidence)
        expanded.update(
            {
                "source_path": block.source_path,
                "line_start": block.line_start,
                "line_end": block.line_end,
                "block_char_start": char_start,
                "block_char_end": char_start + len(evidence["quote"]),
                "quote_sha256": hashlib.sha256(evidence["quote"].encode("utf-8")).hexdigest(),
            }
        )
        expanded_evidence.append(expanded)
    value["evidence"] = expanded_evidence
    value["provenance"] = {
        "schema_version": SCHEMA_VERSION,
        "document_id": document.document_id,
        "document_sha256": document.source_sha256,
        "chunk_id": chunk.chunk_id,
        "actor_local_id": actor_local_id,
        "actor_round": actor_round,
        "actor_model": actor_model,
        "critic_model": critic_model,
        "critic_verdict": "pass",
    }
    value["constraint_id"] = constraint_identifier(value)
    return value


def merge_constraints(
    constraints: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    merged, deduplication_stats = deduplicate_constraints(constraints)
    conflict_groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for constraint in merged:
        conflict_groups[conflict_key(constraint)].append(constraint)

    conflicted_ids: set[str] = set()
    review_items: list[dict[str, Any]] = []
    for key, group in conflict_groups.items():
        assertions = {
            (
                item["assertion"]["polarity"],
                item["assertion"]["operator"],
                _normalized(item["assertion"]["value"]),
                _normalized(item["assertion"]["unit"]),
            )
            for item in group
        }
        if len(group) < 2 or len(assertions) < 2:
            continue
        conflict_id = f"CF-{_hash(key)}"
        for item in group:
            conflicted_ids.add(item["constraint_id"])
        review_items.append(
            {
                "review_type": "potential_cross_chunk_conflict",
                "review_id": conflict_id,
                "message": "Same disease/target/context has differing assertions; verify subtype or missing context before acceptance.",
                "constraints": copy.deepcopy(group),
            }
        )

    accepted = [item for item in merged if item["constraint_id"] not in conflicted_ids]
    accepted.sort(
        key=lambda item: (
            item["provenance"]["document_id"],
            item["constraint_type"],
            item.get("constraint_strength", "strong"),
            item["disease"]["name"].casefold(),
            item["target"]["name"].casefold(),
            item["constraint_id"],
        )
    )
    stats = {
        **deduplication_stats,
        "potential_conflict_groups": len(review_items),
        "accepted_constraints": len(accepted),
    }
    return accepted, review_items, stats
