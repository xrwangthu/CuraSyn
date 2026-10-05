"""Crash-safe incremental checkpoints for Curasyn generation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class GenerationCheckpoint:
    state: dict[str, Any]
    raw_dataframe: pd.DataFrame
    accepted_dataframe: pd.DataFrame
    traces: list[dict[str, Any]]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Checkpoint value is not JSON serializable: {type(value).__name__}")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, default=_json_default)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_checkpoint_round(path: str | Path, payload: dict[str, Any]) -> None:
    journal = Path(path)
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                default=_json_default,
            )
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_checkpoint_state(path: str | Path, payload: dict[str, Any]) -> None:
    _atomic_write_json(Path(path), {"version": CHECKPOINT_VERSION, **payload})


def load_generation_checkpoint(
    state_path: str | Path,
    journal_path: str | Path,
    *,
    expected_identity: str,
    columns: list[str],
) -> GenerationCheckpoint | None:
    state_file = Path(state_path)
    journal_file = Path(journal_path)
    if not state_file.exists() and not journal_file.exists():
        return None
    if not state_file.is_file() or not journal_file.is_file():
        raise RuntimeError("Generation checkpoint state and journal must both exist")
    state = json.loads(state_file.read_text(encoding="utf-8"))
    if state.get("version") != CHECKPOINT_VERSION:
        raise RuntimeError("Unsupported generation checkpoint version")
    if state.get("identity") != expected_identity:
        raise RuntimeError(
            "Generation checkpoint does not match the current data or generation settings"
        )
    completed = int(state.get("rounds_completed", 0))
    entries: dict[int, dict[str, Any]] = {}
    lines = journal_file.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue
            raise RuntimeError("Generation checkpoint journal is corrupt")
        round_number = int(entry["round"])
        if round_number <= completed:
            entries[round_number] = entry
    expected_rounds = list(range(1, completed + 1))
    if sorted(entries) != expected_rounds:
        raise RuntimeError("Generation checkpoint journal has missing completed rounds")

    raw_records: list[dict[str, Any]] = []
    accepted_records: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for round_number in expected_rounds:
        entry = entries[round_number]
        raw_records.extend(entry.get("raw_records", []))
        accepted_records.extend(entry.get("accepted_records", []))
        traces.append(entry["trace"])
    if len(raw_records) != int(state.get("raw_valid_rows", 0)):
        raise RuntimeError("Generation checkpoint raw-row accounting is inconsistent")
    if len(accepted_records) != int(state.get("accepted_rows", 0)):
        raise RuntimeError("Generation checkpoint accepted-row accounting is inconsistent")
    return GenerationCheckpoint(
        state=state,
        raw_dataframe=pd.DataFrame(raw_records, columns=columns),
        accepted_dataframe=pd.DataFrame(accepted_records, columns=columns),
        traces=traces,
    )


def clear_generation_checkpoint(state_path: str | Path, journal_path: str | Path) -> None:
    Path(state_path).unlink(missing_ok=True)
    Path(journal_path).unlink(missing_ok=True)
