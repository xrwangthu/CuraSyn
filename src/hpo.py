"""Load the frozen Human Phenotype Ontology used in PID prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HPO_PATH = ROOT / "constraintKB" / "ontology" / "hp.obo"
HPO_CODE_PATTERN = re.compile(r"HP:\d{7}")


@dataclass(frozen=True)
class HpoOntology:
    version: str
    names: dict[str, str]
    canonical_ids: dict[str, str]


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute() and not resolved.is_file():
        resolved = ROOT / resolved
    if not resolved.is_file():
        raise FileNotFoundError(f"HPO ontology does not exist: {resolved}")
    return resolved


@lru_cache(maxsize=None)
def load_hpo_ontology(path: str | Path = DEFAULT_HPO_PATH) -> HpoOntology:
    """Parse active terms and aliases from a pinned OBO release."""
    resolved = _resolve_path(path)
    version = ""
    terms: list[dict[str, object]] = []
    current: dict[str, object] | None = None

    with resolved.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not version and line.startswith("data-version:"):
                version = line.split(":", 1)[1].strip().rsplit("/", 1)[-1]
            if line == "[Term]":
                if current is not None:
                    terms.append(current)
                current = {"alt_ids": [], "obsolete": False}
                continue
            if line.startswith("["):
                if current is not None:
                    terms.append(current)
                    current = None
                continue
            if current is None:
                continue
            if line.startswith("id: "):
                current["id"] = line[4:]
            elif line.startswith("name: "):
                current["name"] = line[6:]
            elif line.startswith("alt_id: "):
                current["alt_ids"].append(line[8:])
            elif line == "is_obsolete: true":
                current["obsolete"] = True
    if current is not None:
        terms.append(current)

    if not version:
        raise ValueError(f"HPO ontology has no data-version: {resolved}")

    active_names: dict[str, str] = {}
    obsolete_names: dict[str, str] = {}
    active_aliases: dict[str, tuple[str, str]] = {}
    for term in terms:
        code = term.get("id")
        name = term.get("name")
        if (
            not isinstance(code, str)
            or HPO_CODE_PATTERN.fullmatch(code) is None
            or not isinstance(name, str)
            or not name
        ):
            continue
        if term["obsolete"]:
            obsolete_names[code] = name
            continue
        active_names[code] = name
        for alias in term["alt_ids"]:
            if isinstance(alias, str) and HPO_CODE_PATTERN.fullmatch(alias):
                active_aliases[alias] = (code, name)

    names = dict(obsolete_names)
    canonical_ids = {code: code for code in obsolete_names}
    names.update(active_names)
    canonical_ids.update({code: code for code in active_names})
    for alias, (canonical_id, name) in active_aliases.items():
        names[alias] = name
        canonical_ids[alias] = canonical_id

    if not active_names:
        raise ValueError(f"HPO ontology contains no active terms: {resolved}")
    return HpoOntology(version, names, canonical_ids)


def require_hpo_names(
    codes: Iterable[str],
    ontology: HpoOntology,
) -> dict[str, str]:
    """Return names for a vocabulary or fail before generation starts."""
    unique_codes = sorted(set(codes))
    missing = [code for code in unique_codes if code not in ontology.names]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(
            f"HPO ontology {ontology.version} has no name for "
            f"{len(missing)} code(s): {preview}{suffix}"
        )
    return {code: ontology.names[code] for code in unique_codes}
