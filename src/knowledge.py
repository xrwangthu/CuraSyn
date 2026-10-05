"""Read-only retrieval for clinical background knowledge used in generation."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
RELATION_TO_TYPE = {
    "associated_with_phenotype": "phenotype",
    "associated_with_examination": "examination",
    "associated_with_demographic": "demographic",
}


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class KnowledgeRetrieval:
    disease_names: list[str]
    matched_diseases: list[dict[str, str]]
    knowledge: list[dict[str, Any]]
    unmatched_disease_names: list[str]
    kb_checksum: str | None

    @property
    def knowledge_ids(self) -> list[str]:
        return [item["knowledge_id"] for item in self.knowledge]

    def trace(self) -> dict[str, Any]:
        trace = {
            "query_disease_names": self.disease_names,
            "matched_diseases": self.matched_diseases,
            "unmatched_disease_names": self.unmatched_disease_names,
            "retrieved_knowledge_ids": self.knowledge_ids,
        }
        empirical = [
            {
                "knowledge_id": item["knowledge_id"],
                "target_code": item["target_code"],
                "frequency": float(item["empirical_frequency"]),
                **(
                    {
                        "residual_probability": float(
                            item["residual_probability"]
                        )
                    }
                    if item.get("residual_probability") is not None
                    else {}
                ),
            }
            for item in self.knowledge
            if item.get("empirical_frequency") is not None
        ]
        if empirical:
            trace["retrieved_empirical_hpo"] = empirical
        return trace


EMPIRICAL_HPO_COLUMNS = ("disease", "hpo", "frequency")


def mine_empirical_hpo_relations(
    train_df: Any,
    schema: dict[str, Any],
    minimum_support: int,
) -> Any:
    """Mine deterministic record-level disease--HPO frequencies from D_train."""
    if isinstance(minimum_support, bool) or not isinstance(minimum_support, int):
        raise ValueError("knowledge.empirical_hpo.minimum_support must be an integer")
    if minimum_support <= 0:
        raise ValueError("knowledge.empirical_hpo.minimum_support must be positive")
    disease_column = schema["disease_name_column"]
    phenotype_column = schema["phenotype_column"]
    missing = sorted({disease_column, phenotype_column} - set(train_df.columns))
    if missing:
        raise ValueError(f"D_train is missing empirical HPO columns: {missing}")

    from src.data import parse_hpo_codes
    import pandas as pd

    disease_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    for disease_value, hpo_value in train_df[
        [disease_column, phenotype_column]
    ].itertuples(index=False, name=None):
        diseases = set(parse_disease_names([disease_value]))
        hpo_codes = set(parse_hpo_codes(hpo_value, schema))
        disease_counts.update(diseases)
        pair_counts.update(
            (disease, code)
            for disease in diseases
            for code in hpo_codes
        )

    records = [
        {
            "disease": disease,
            "hpo": code,
            "frequency": count / disease_counts[disease],
        }
        for (disease, code), count in pair_counts.items()
        if count >= minimum_support
    ]
    records.sort(key=lambda row: (row["disease"], -row["frequency"], row["hpo"]))
    return pd.DataFrame.from_records(records, columns=EMPIRICAL_HPO_COLUMNS)


class EmpiricalHpoKnowledge:
    """In-memory, dataset-specific overlay; the general SQLite remains immutable."""

    def __init__(self, relations: Any):
        if tuple(relations.columns) != EMPIRICAL_HPO_COLUMNS:
            raise ValueError(
                f"Empirical HPO columns must be {list(EMPIRICAL_HPO_COLUMNS)}"
            )
        self.relations = relations.copy()
        self._by_disease = {
            disease: group.reset_index(drop=True)
            for disease, group in self.relations.groupby("disease", sort=True)
        }

    def retrieve(self, disease_names: Iterable[str], max_items: int) -> list[dict[str, Any]]:
        if max_items < 0:
            raise ValueError("max_items must be non-negative")
        output: list[dict[str, Any]] = []
        for disease in dict.fromkeys(str(value).strip() for value in disease_names):
            frame = self._by_disease.get(disease)
            if frame is None:
                continue
            output.extend(frame.head(max_items).to_dict("records"))
        return output


def merge_empirical_hpo_knowledge(
    retrieval: KnowledgeRetrieval,
    empirical: EmpiricalHpoKnowledge | None,
    hpo_names: dict[str, str],
    *,
    max_items: int,
) -> KnowledgeRetrieval:
    """Merge one train-derived HPO line per code into general retrieval."""
    if empirical is None:
        return retrieval
    relations = empirical.retrieve(retrieval.disease_names, max_items)
    if not relations:
        return retrieval

    items = [dict(item) for item in retrieval.knowledge]
    for item in items:
        item.setdefault("knowledge_source", "general")
    for relation in relations:
        code = relation["hpo"]
        frequency = float(relation["frequency"])
        matches = [
            index for index, item in enumerate(items)
            if item["target_type"] == "phenotype" and item.get("target_code") == code
        ]
        if matches:
            primary = matches[0]
            items[primary]["empirical_frequency"] = frequency
            items[primary]["knowledge_source"] = "general_and_empirical"
            for duplicate in reversed(matches[1:]):
                del items[duplicate]
            continue
        disease = relation["disease"]
        identifier = hashlib.sha1(
            f"{disease}\x1f{code}".encode("utf-8")
        ).hexdigest()[:16]
        name = hpo_names[code]
        items.append(
            {
                "knowledge_id": f"EMPIRICAL_HPO:{identifier}",
                "disease_name": disease,
                "target_type": "phenotype",
                "target_name": name,
                "target_code": code,
                "statement": f"{name} ({code})",
                "strength": "",
                "source_constraint_id": "train_empirical",
                "match_scope": "exact_train_disease",
                "knowledge_source": "empirical",
                "empirical_frequency": frequency,
            }
        )

    matched = list(retrieval.matched_diseases)
    matched_queries = {item["query_name"] for item in matched}
    empirical_diseases = {relation["disease"] for relation in relations}
    for disease in retrieval.disease_names:
        if disease in empirical_diseases and disease not in matched_queries:
            matched.append(
                {
                    "query_name": disease,
                    "node_id": f"EMPIRICAL_DISEASE:{hashlib.sha1(disease.encode()).hexdigest()[:16]}",
                    "canonical_name": disease,
                    "match_scope": "exact_train_disease",
                }
            )
    return KnowledgeRetrieval(
        retrieval.disease_names,
        matched,
        items,
        [name for name in retrieval.unmatched_disease_names if name not in empirical_diseases],
        retrieval.kb_checksum,
    )


def attach_empirical_hpo_residual_probabilities(
    retrieval: KnowledgeRetrieval,
    probabilities: dict[str, float],
) -> KnowledgeRetrieval:
    """Attach dynamic probabilities only to train-derived disease--HPO edges."""
    items: list[dict[str, Any]] = []
    for original in retrieval.knowledge:
        item = dict(original)
        code = item.get("target_code")
        if item.get("empirical_frequency") is not None and code in probabilities:
            probability = float(probabilities[code])
            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"Residual HPO probability for {code} must be in [0, 1]"
                )
            item["residual_probability"] = probability
        items.append(item)
    return KnowledgeRetrieval(
        retrieval.disease_names,
        list(retrieval.matched_diseases),
        items,
        list(retrieval.unmatched_disease_names),
        retrieval.kb_checksum,
    )


class ClinicalKnowledgeBase:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        if self.path is not None and not self.path.is_absolute() and not self.path.is_file():
            self.path = ROOT / self.path
        if self.path is not None:
            if not self.path.is_file():
                raise FileNotFoundError(f"Knowledge base does not exist: {self.path}")
            self.checksum = _sha256(self.path)
            with self._connect() as connection:
                version = connection.execute(
                    "SELECT value FROM metadata WHERE key = 'schema_version'"
                ).fetchone()
                if version is None or version[0] != "1":
                    raise ValueError("Unsupported clinical knowledge database schema")
        else:
            self.checksum = None

    def _connect(self) -> sqlite3.Connection:
        if self.path is None:
            raise RuntimeError("The empty knowledge base has no database connection")
        connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def retrieve(
        self,
        disease_names: Iterable[str],
        *,
        max_items: int = 12,
        max_phenotype_items: int = 8,
        max_examination_items: int = 4,
        include_demographic: bool = True,
    ) -> KnowledgeRetrieval:
        names = list(dict.fromkeys(str(name).strip() for name in disease_names if str(name).strip()))
        if max_items <= 0 or max_phenotype_items < 0 or max_examination_items < 0:
            raise ValueError("Knowledge retrieval limits must be non-negative and max_items positive")
        if self.path is None or not names:
            return KnowledgeRetrieval(names, [], [], names, self.checksum)

        matched: list[dict[str, str]] = []
        unmatched: list[str] = []
        disease_ids: list[str] = []
        with self._connect() as connection:
            for name in names:
                rows = connection.execute(
                    """
                    SELECT n.node_id, n.canonical_name
                    FROM aliases a JOIN nodes n ON n.node_id = a.node_id
                    WHERE a.normalized_alias = ? AND n.node_type = 'disease'
                    ORDER BY n.canonical_name
                    """,
                    (normalize_name(name),),
                ).fetchall()
                if len(rows) != 1:
                    unmatched.append(name)
                    continue
                row = rows[0]
                disease_ids.append(row["node_id"])
                matched.append(
                    {
                        "query_name": name,
                        "node_id": row["node_id"],
                        "canonical_name": row["canonical_name"],
                        "match_scope": "exact_or_confirmed_alias",
                    }
                )

            items: list[dict[str, str]] = []
            for disease_id in dict.fromkeys(disease_ids):
                rows = connection.execute(
                    """
                    SELECT e.edge_id, e.relation_type, e.statement, e.strength_text,
                           e.source_constraint_id, target.canonical_name AS target_name,
                           target.standard_code AS target_code,
                           disease.canonical_name AS disease_name
                    FROM edges e
                    JOIN nodes target ON target.node_id = e.target_node_id
                    JOIN nodes disease ON disease.node_id = e.source_node_id
                    WHERE e.source_node_id = ?
                      AND e.relation_type IN (
                        'associated_with_phenotype',
                        'associated_with_examination',
                        'associated_with_demographic'
                      )
                    ORDER BY e.source_constraint_id, e.edge_id
                    """,
                    (disease_id,),
                ).fetchall()
                for row in rows:
                    target_type = RELATION_TO_TYPE[row["relation_type"]]
                    if target_type == "demographic" and not include_demographic:
                        continue
                    items.append(
                        {
                            "knowledge_id": row["edge_id"],
                            "disease_name": row["disease_name"],
                            "target_type": target_type,
                            "target_name": row["target_name"],
                            "target_code": row["target_code"],
                            "statement": row["statement"],
                            "strength": row["strength_text"],
                            "source_constraint_id": row["source_constraint_id"],
                            "match_scope": "exact_disease",
                        }
                    )

        deduplicated: list[dict[str, str]] = []
        seen: set[tuple[str, str, str, str]] = set()
        limits = {
            "phenotype": max_phenotype_items,
            "examination": max_examination_items,
            "demographic": max_items,
        }
        counts = {key: 0 for key in limits}
        for item in items:
            key = (
                item["disease_name"],
                item["target_code"] or item["target_name"],
                item["statement"],
                item["strength"],
            )
            target_type = item["target_type"]
            if key in seen or counts[target_type] >= limits[target_type]:
                continue
            seen.add(key)
            counts[target_type] += 1
            deduplicated.append(item)
            if len(deduplicated) >= max_items:
                break
        return KnowledgeRetrieval(names, matched, deduplicated, unmatched, self.checksum)


def filter_phenotype_knowledge_to_vocabulary(
    retrieval: KnowledgeRetrieval,
    allowed_hpo_codes: Iterable[str],
    canonical_ids: dict[str, str],
) -> tuple[KnowledgeRetrieval, list[str]]:
    """Remove phenotype knowledge that the dataset cannot represent."""
    allowed = set(allowed_hpo_codes)
    kept: list[dict[str, str]] = []
    filtered_ids: list[str] = []
    for item in retrieval.knowledge:
        if item["target_type"] != "phenotype":
            kept.append(item)
            continue
        code = item.get("target_code")
        canonical = canonical_ids.get(code, code) if code else None
        if code in allowed or canonical in allowed:
            kept.append(item)
        else:
            filtered_ids.append(item["knowledge_id"])
    return (
        KnowledgeRetrieval(
            retrieval.disease_names,
            retrieval.matched_diseases,
            kept,
            retrieval.unmatched_disease_names,
            retrieval.kb_checksum,
        ),
        filtered_ids,
    )


def parse_disease_names(values: Iterable[Any]) -> list[str]:
    names: list[str] = []
    for value in values:
        if value is None or (isinstance(value, float) and value != value):
            continue
        if isinstance(value, list):
            parsed = value
        elif isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = [value]
        else:
            continue
        if isinstance(parsed, list):
            names.extend(str(name).strip() for name in parsed if isinstance(name, str) and name.strip())
    return list(dict.fromkeys(names))


def render_knowledge(retrieval: KnowledgeRetrieval) -> str:
    if not retrieval.knowledge:
        return "No disease-specific background knowledge was retrieved."
    headings = {
        "phenotype": "Phenotypes",
        "examination": "Examinations",
        "demographic": "Demographics",
    }
    lines = ["Retrieved clinical background knowledge:"]
    if retrieval.matched_diseases:
        names = list(dict.fromkeys(
            item["canonical_name"] for item in retrieval.matched_diseases
        ))
        lines.append(f"Disease: {'; '.join(names)}")
    for target_type in ("phenotype", "examination", "demographic"):
        items = [item for item in retrieval.knowledge if item["target_type"] == target_type]
        if not items:
            continue
        lines.append(f"{headings[target_type]}:")
        for item in items:
            frequency = item.get("empirical_frequency")
            if target_type == "phenotype" and frequency is not None:
                line = (
                    f"- {item['target_name']} ({item['target_code']}): observed in "
                    f"approximately {100 * float(frequency):.1f}% of D_train records "
                    f"with this disease."
                )
                residual_probability = item.get("residual_probability")
                if residual_probability is not None:
                    line += (
                        " Recommended inclusion probability among the remaining "
                        "accepted records: approximately "
                        f"{100 * float(residual_probability):.1f}%."
                    )
                if item["strength"]:
                    line += f" General evidence description: {item['strength']}."
                lines.append(line)
                continue
            qualifier = f" Evidence description: {item['strength']}." if item["strength"] else ""
            lines.append(f"- {item['statement']}.{qualifier}")
    lines.append("Use these observations only as clinical background.")
    if any(item.get("empirical_frequency") is not None for item in retrieval.knowledge):
        lines.append("D_train frequencies are aggregate observations, not per-record quotas.")
    if any(item.get("residual_probability") is not None for item in retrieval.knowledge):
        lines.append(
            "Remaining-record probabilities are dynamic aggregate targets, not "
            "mandatory per-record conditions."
        )
    lines.append(
        "They are not mandatory conditions; do not force every item into every generated patient."
    )
    return "\n".join(lines)
