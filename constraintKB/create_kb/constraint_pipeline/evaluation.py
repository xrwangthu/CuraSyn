from __future__ import annotations

import math
from typing import Any

from .merge import semantic_key


def _wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _score(predicted: set[tuple[str, ...]], gold: set[tuple[str, ...]]) -> dict[str, Any]:
    true_positive = len(predicted & gold)
    false_positive = len(predicted - gold)
    false_negative = len(gold - predicted)
    precision = true_positive / len(predicted) if predicted else (1.0 if not gold else 0.0)
    recall = true_positive / len(gold) if gold else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "precision_wilson_95": _wilson(true_positive, len(predicted)),
        "recall": recall,
        "recall_wilson_95": _wilson(true_positive, len(gold)),
        "f1": f1,
    }


def evaluate_constraints(
    predictions: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    *,
    min_precision: float = 0.0,
    min_recall: float = 0.0,
) -> dict[str, Any]:
    predicted_by_key = {semantic_key(item): item for item in predictions}
    gold_by_key = {semantic_key(item): item for item in gold}
    predicted_keys = set(predicted_by_key)
    gold_keys = set(gold_by_key)
    overall = _score(predicted_keys, gold_keys)
    per_type: dict[str, Any] = {}
    all_types = sorted({key[0] for key in predicted_keys | gold_keys})
    for constraint_type in all_types:
        per_type[constraint_type] = _score(
            {key for key in predicted_keys if key[0] == constraint_type},
            {key for key in gold_keys if key[0] == constraint_type},
        )
    passed = overall["precision"] >= min_precision and overall["recall"] >= min_recall
    return {
        "match_definition": "normalized exact semantic key; evidence and provenance are ignored",
        "prediction_rows": len(predictions),
        "unique_predictions": len(predicted_keys),
        "gold_rows": len(gold),
        "unique_gold": len(gold_keys),
        "overall": overall,
        "per_type": per_type,
        "thresholds": {"min_precision": min_precision, "min_recall": min_recall},
        "passed_thresholds": passed,
        "false_positive_examples": [
            predicted_by_key[key] for key in sorted(predicted_keys - gold_keys)[:20]
        ],
        "false_negative_examples": [
            gold_by_key[key] for key in sorted(gold_keys - predicted_keys)[:20]
        ],
    }
