"""Build the Curasyn clinical-background knowledge graph in SQLite."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "data" / "constraints.json"
DEFAULT_MAPPING = ROOT / "disease_mapping.csv"
DEFAULT_OUTPUT = ROOT / "kb" / "clinical_knowledge.sqlite"
HPO_PATTERN = re.compile(r"^HPO:(\d{7})\s*-\s*(.+)$")

EXAMINATION_PATTERNS = (
    (r"\babsolute neutrophil count\b|\banc\b", "examination__anc", "ANC"),
    (r"\babsolute lymphocyte count\b|\balc\b", "examination__alc", "ALC"),
    (r"\bplatelet(?:s| count)?\b", "examination__platelets", "Platelets"),
    (r"\bhemoglobin\b", "examination__hemoglobin", "Hemoglobin"),
    (r"\bserum sodium\b|\bsodium\b", "examination__sodium", "Sodium"),
    (r"\baspartate aminotransferase\b|\bast\b", "examination__ast", "AST"),
    (r"\balanine aminotransferase\b|\balt\b", "examination__alt", "ALT"),
    (r"\btotal bilirubin\b|\bbilirubin\b", "examination__bilirubin_total", "Total bilirubin"),
    (r"\bcreatinine\b", "examination__creatinine", "Creatinine"),
    (r"\bprothrombin time\b", "examination__pt", "Prothrombin time"),
    (r"\b(?:serum )?igg\b", "examination__pr_igg", "IgG"),
    (r"\b(?:serum )?igm\b", "examination__pr_igm", "IgM"),
)


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def stable_id(prefix: str, *parts: str) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return f"{prefix}:{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]}"


def load_mappings(path: Path) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"canonical_name", "iuis_table", "aliases"}
    if set(frame.columns) != required:
        raise ValueError(f"Disease mapping columns must be {sorted(required)}")
    aliases: dict[str, dict[str, str]] = {}
    rows: list[dict[str, str]] = []
    for row in frame.to_dict("records"):
        names = [row["canonical_name"], *row["aliases"].split("|")]
        for name in names:
            key = normalize_name(name)
            prior = aliases.get(key)
            if prior is not None and prior["canonical_name"] != row["canonical_name"]:
                raise ValueError(f"Ambiguous disease alias: {name!r}")
            aliases[key] = row
        rows.append(row)
    return aliases, rows


def examination_target(statement: str) -> tuple[str, str]:
    normalized = normalize_name(statement)
    for pattern, column, display_name in EXAMINATION_PATTERNS:
        if re.search(pattern, normalized):
            return column, display_name
    label = re.split(r"(?:<=|>=|<|>|=|\d)", statement, maxsplit=1)[0].strip()
    return "", label or statement


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE nodes (
            node_id TEXT PRIMARY KEY,
            node_type TEXT NOT NULL CHECK (
                node_type IN ('disease', 'phenotype', 'examination', 'demographic', 'category')
            ),
            canonical_name TEXT NOT NULL,
            standard_code TEXT NOT NULL DEFAULT '',
            UNIQUE (node_type, canonical_name, standard_code)
        );
        CREATE TABLE aliases (
            normalized_alias TEXT NOT NULL,
            alias TEXT NOT NULL,
            node_id TEXT NOT NULL REFERENCES nodes(node_id),
            PRIMARY KEY (normalized_alias, node_id)
        );
        CREATE TABLE edges (
            edge_id TEXT PRIMARY KEY,
            source_node_id TEXT NOT NULL REFERENCES nodes(node_id),
            relation_type TEXT NOT NULL CHECK (
                relation_type IN (
                    'associated_with_phenotype',
                    'associated_with_examination',
                    'associated_with_demographic',
                    'belongs_to_table'
                )
            ),
            target_node_id TEXT NOT NULL REFERENCES nodes(node_id),
            statement TEXT NOT NULL,
            strength_text TEXT NOT NULL,
            source_constraint_id TEXT NOT NULL
        );
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE INDEX idx_alias_lookup ON aliases(normalized_alias);
        CREATE INDEX idx_edge_source ON edges(source_node_id, relation_type);
        CREATE INDEX idx_edge_target ON edges(target_node_id);
        """
    )


def insert_node(
    connection: sqlite3.Connection,
    node_type: str,
    canonical_name: str,
    standard_code: str = "",
) -> str:
    node_id = stable_id(node_type.upper(), canonical_name, standard_code)
    connection.execute(
        "INSERT OR IGNORE INTO nodes VALUES (?, ?, ?, ?)",
        (node_id, node_type, canonical_name, standard_code),
    )
    return node_id


def insert_alias(connection: sqlite3.Connection, alias: str, node_id: str) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO aliases VALUES (?, ?, ?)",
        (normalize_name(alias), alias, node_id),
    )


def insert_edge(
    connection: sqlite3.Connection,
    source_node_id: str,
    relation_type: str,
    target_node_id: str,
    statement: str,
    strength: str,
    source_constraint_id: str,
    ordinal: int,
) -> None:
    edge_id = stable_id(
        "EDGE", source_constraint_id, str(ordinal), source_node_id,
        relation_type, target_node_id, statement, strength,
    )
    connection.execute(
        "INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            edge_id, source_node_id, relation_type, target_node_id,
            statement, strength, source_constraint_id,
        ),
    )


def build_database(source: Path, mapping: Path, output: Path) -> dict[str, int]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    constraints = payload.get("constraints")
    if not isinstance(constraints, dict):
        raise ValueError("constraints.json must contain a constraints object")
    alias_mapping, mapping_rows = load_mappings(mapping)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        create_schema(connection)
        canonical_nodes: dict[str, str] = {}
        for row in sorted(mapping_rows, key=lambda item: item["canonical_name"]):
            disease_id = insert_node(connection, "disease", row["canonical_name"])
            canonical_nodes[row["canonical_name"]] = disease_id
            for alias in [row["canonical_name"], *row["aliases"].split("|")]:
                insert_alias(connection, alias, disease_id)
            table_id = insert_node(connection, "category", row["iuis_table"])
            insert_edge(
                connection, disease_id, "belongs_to_table", table_id,
                row["iuis_table"], "", "disease_mapping", 0,
            )

        for source_id, record in sorted(constraints.items(), key=lambda item: int(item[0])):
            if not isinstance(record, dict) or not isinstance(record.get("IF"), str):
                raise ValueError(f"Invalid constraint record {source_id}")
            raw_disease = record["IF"].strip()
            mapped = alias_mapping.get(normalize_name(raw_disease))
            canonical = mapped["canonical_name"] if mapped else raw_disease
            disease_id = canonical_nodes.get(canonical)
            if disease_id is None:
                disease_id = insert_node(connection, "disease", canonical)
                canonical_nodes[canonical] = disease_id
            insert_alias(connection, raw_disease, disease_id)

            ordinal = 0
            for item in record.get("HPO", []):
                ordinal += 1
                match = HPO_PATTERN.fullmatch(str(item.get("CONSTRAINT", "")).strip())
                if match is None:
                    raise ValueError(f"Invalid HPO constraint in {source_id}: {item}")
                code, name = f"HP:{match.group(1)}", match.group(2).strip()
                target_id = insert_node(connection, "phenotype", name, code)
                statement = f"{name} ({code})"
                insert_edge(
                    connection, disease_id, "associated_with_phenotype", target_id,
                    statement, str(item.get("STRENGTH", "")).strip(), source_id, ordinal,
                )

            for item in record.get("THEN", []):
                ordinal += 1
                statement = str(item.get("CONSTRAINT", "")).strip()
                column, name = examination_target(statement)
                target_id = insert_node(connection, "examination", name, column)
                insert_edge(
                    connection, disease_id, "associated_with_examination", target_id,
                    statement, str(item.get("STRENGTH", "")).strip(), source_id, ordinal,
                )

        source_checksum = hashlib.sha256(source.read_bytes()).hexdigest()
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [
                ("schema_version", "1"),
                ("source_checksum", source_checksum),
                ("purpose", "llm_generation_background_knowledge"),
            ],
        )
        connection.commit()
        counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("nodes", "aliases", "edges")
        }
        counts["unmapped_examination_nodes"] = int(connection.execute(
            "SELECT COUNT(*) FROM nodes WHERE node_type = 'examination' AND standard_code = ''"
        ).fetchone()[0])
    finally:
        connection.close()
    temporary.replace(output)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = build_database(args.source, args.mapping, args.output)
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
