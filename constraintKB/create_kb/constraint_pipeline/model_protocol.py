from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from typing import Any

from .markdown import Chunk


@dataclass(slots=True)
class BlockReferenceCodec:
    """Translate compact model-facing block aliases to auditable source IDs."""

    full_to_alias: dict[str, str]
    alias_to_full: dict[str, str]
    legacy_to_full: dict[str, str]

    @classmethod
    def for_chunk(cls, chunk: Chunk) -> "BlockReferenceCodec":
        full_to_alias: dict[str, str] = {}
        for index, block_id in enumerate(chunk.context_block_ids, 1):
            full_to_alias[block_id] = f"C{index:03d}"
        for index, block_id in enumerate(chunk.primary_block_ids, 1):
            full_to_alias[block_id] = f"P{index:03d}"

        legacy_candidates: dict[str, set[str]] = {}
        for block_id in chunk.all_block_ids:
            parts = block_id.rsplit(":", 2)
            forms = {block_id}
            if len(parts) == 3:
                block_number, digest = parts[-2:]
                forms.update(
                    {
                        block_number,
                        digest,
                        f":{digest}",
                        f"{block_number}:{digest}",
                        f":{block_number}:{digest}",
                    }
                )
            for form in forms:
                legacy_candidates.setdefault(form, set()).add(block_id)

        return cls(
            full_to_alias=full_to_alias,
            alias_to_full={alias: full for full, alias in full_to_alias.items()},
            legacy_to_full={
                reference: next(iter(matches))
                for reference, matches in legacy_candidates.items()
                if len(matches) == 1
            },
        )

    def model_chunk(self, chunk: Chunk) -> Chunk:
        text = chunk.text
        for block_id, alias in sorted(
            self.full_to_alias.items(), key=lambda item: len(item[0]), reverse=True
        ):
            text = text.replace(block_id, alias)
        return replace(
            chunk,
            primary_block_ids=[self.full_to_alias[item] for item in chunk.primary_block_ids],
            context_block_ids=[self.full_to_alias[item] for item in chunk.context_block_ids],
            text=text,
            char_count=len(text),
        )

    def expand_actor_result(self, result: dict[str, Any]) -> dict[str, Any]:
        value = copy.deepcopy(result)
        reviewed = value.get("reviewed_primary_block_ids")
        if isinstance(reviewed, list):
            value["reviewed_primary_block_ids"] = [
                self._expand_reference(item) for item in reviewed
            ]
        constraints = value.get("constraints")
        if isinstance(constraints, list):
            for candidate in constraints:
                if not isinstance(candidate, dict):
                    continue
                evidence = candidate.get("evidence")
                if not isinstance(evidence, list):
                    continue
                for item in evidence:
                    if isinstance(item, dict) and "block_id" in item:
                        item["block_id"] = self._expand_reference(item["block_id"])
        return value

    def compact_actor_result(self, result: dict[str, Any] | None) -> dict[str, Any] | None:
        if result is None:
            return None
        value = copy.deepcopy(result)
        reviewed = value.get("reviewed_primary_block_ids")
        if isinstance(reviewed, list):
            value["reviewed_primary_block_ids"] = [
                self._compact_reference(item) for item in reviewed
            ]
        constraints = value.get("constraints")
        if isinstance(constraints, list):
            for candidate in constraints:
                if not isinstance(candidate, dict):
                    continue
                evidence = candidate.get("evidence")
                if not isinstance(evidence, list):
                    continue
                for item in evidence:
                    if isinstance(item, dict) and "block_id" in item:
                        item["block_id"] = self._compact_reference(item["block_id"])
        return value

    def expand_critic_result(self, result: dict[str, Any]) -> dict[str, Any]:
        value = copy.deepcopy(result)
        reviewed = value.get("reviewed_primary_block_ids")
        if isinstance(reviewed, list):
            value["reviewed_primary_block_ids"] = [
                self._expand_reference(item) for item in reviewed
            ]
        issues = value.get("issues")
        if isinstance(issues, list):
            for item in issues:
                if isinstance(item, dict) and item.get("block_id") is not None:
                    item["block_id"] = self._expand_reference(item["block_id"])
        return value

    def compact_critic_result(self, result: dict[str, Any]) -> dict[str, Any]:
        value = copy.deepcopy(result)
        reviewed = value.get("reviewed_primary_block_ids")
        if isinstance(reviewed, list):
            value["reviewed_primary_block_ids"] = [
                self._compact_reference(item) for item in reviewed
            ]
        issues = value.get("issues")
        if isinstance(issues, list):
            for item in issues:
                if isinstance(item, dict) and item.get("block_id") is not None:
                    item["block_id"] = self._compact_reference(item["block_id"])
        return value

    def compact_feedback(self, feedback: list[dict[str, Any]]) -> list[dict[str, Any]]:
        value = copy.deepcopy(feedback)
        for item in value:
            if item.get("block_id") is not None:
                item["block_id"] = self._compact_reference(item["block_id"])
            for key in ("message", "required_action"):
                text = item.get(key)
                if not isinstance(text, str):
                    continue
                for block_id, alias in sorted(
                    self.full_to_alias.items(),
                    key=lambda pair: len(pair[0]),
                    reverse=True,
                ):
                    text = text.replace(block_id, alias)
                item[key] = text
        return value

    def compact_revision_feedback(
        self, feedback: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Keep revision feedback actionable without treating critic prose as schema."""

        compacted: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in self.compact_feedback(feedback):
            keys = [
                "issue_id",
                "category",
                "candidate_id",
                "block_id",
                "evidence_quote",
            ]
            if item.get("category") != "missing_constraint":
                keys.append("required_action")
            reduced = {key: item.get(key) for key in keys if key in item}
            fingerprint = json.dumps(
                {key: value for key, value in reduced.items() if key != "issue_id"},
                ensure_ascii=False,
                sort_keys=True,
            )
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            compacted.append(reduced)
        return compacted

    def _expand_reference(self, reference: Any) -> Any:
        if not isinstance(reference, str):
            return reference
        return self.alias_to_full.get(
            reference, self.legacy_to_full.get(reference, reference)
        )

    def _compact_reference(self, reference: Any) -> Any:
        if not isinstance(reference, str):
            return reference
        full = self.alias_to_full.get(
            reference, self.legacy_to_full.get(reference, reference)
        )
        return self.full_to_alias.get(full, reference)
