"""Disease scheduling and exemplar retrieval for the PID generation path."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import special, stats
from scipy.spatial import distance
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample

from src.data import NumericalValidityProfile, parse_hpo_codes
from src.knowledge import parse_disease_names


@dataclass(frozen=True)
class DiseaseSchedule:
    targets: list[tuple[str, int]]
    weights: dict[tuple[str, int], float]
    target_counts: dict[tuple[str, int], int]
    disease_to_labels: dict[str, list[int]]


@dataclass
class ExemplarSelection:
    dataframe: pd.DataFrame
    trace: dict[str, Any]


@dataclass
class ResidualFrame:
    dataframe: pd.DataFrame
    categorical_columns: set[str]


@dataclass(frozen=True)
class PhenotypeProfile:
    codes: list[str]
    source: str
    source_rows: int
    minimum_support: int
    coverage_available: bool
    coverage_source: str
    hpo_count_range: tuple[int, int]
    examination_targets: list[dict[str, Any]]
    missing_count_range: tuple[int, int]
    missing_columns: list[str]
    demographic_targets: list[dict[str, Any]]
    numerical_bounds: dict[str, tuple[float, float] | None]
    joint_support: int | None = None

    def trace(self) -> dict[str, Any]:
        trace = {
            "source": self.source,
            "source_rows": self.source_rows,
            "minimum_support": self.minimum_support,
            "codes": self.codes,
            "coverage_available": self.coverage_available,
            "coverage_source": self.coverage_source,
            "hpo_count_range": list(self.hpo_count_range),
            "examination_targets": self.examination_targets,
            "missing_count_range": list(self.missing_count_range),
            "missing_columns": self.missing_columns,
            "demographic_targets": self.demographic_targets,
        }
        if self.joint_support is not None:
            trace["joint_support"] = self.joint_support
        return trace


def _hpo_counts(
    dataframe: pd.DataFrame,
    schema: dict[str, Any],
) -> Counter[str]:
    phenotype_column = schema["phenotype_column"]
    counts: Counter[str] = Counter()
    for value in dataframe[phenotype_column]:
        counts.update(set(parse_hpo_codes(value, schema)))
    return counts


def residual_empirical_hpo_probabilities(
    empirical_frequencies: dict[str, float],
    accepted_disease: pd.DataFrame,
    schema: dict[str, Any],
    target_rows: int,
) -> dict[str, float]:
    """Return probabilities needed across the remaining disease-level quota."""
    if target_rows <= 0:
        raise ValueError("Disease-level target_rows must be positive")
    accepted_rows = len(accepted_disease)
    if accepted_rows > target_rows:
        raise ValueError("Accepted disease rows exceed the disease-level target")
    remaining_rows = target_rows - accepted_rows
    if remaining_rows == 0:
        return {}
    accepted_counts = _hpo_counts(accepted_disease, schema)
    output: dict[str, float] = {}
    for code, raw_frequency in sorted(empirical_frequencies.items()):
        frequency = float(raw_frequency)
        if not 0.0 <= frequency <= 1.0:
            raise ValueError(
                f"Empirical HPO frequency for {code} must be in [0, 1]"
            )
        residual = (
            target_rows * frequency - accepted_counts[code]
        ) / remaining_rows
        output[code] = float(np.clip(residual, 0.0, 1.0))
    return output


def select_frequency_aligned_candidates(
    candidates: pd.DataFrame,
    schema: dict[str, Any],
    probabilities: dict[str, float],
    n_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Choose whole passed records whose marginal HPO rates best match targets."""
    if n_rows < 0:
        raise ValueError("n_rows must be non-negative")
    take = min(n_rows, len(candidates))
    codes = sorted(probabilities)
    trace: dict[str, Any] = {
        "applied": False,
        "candidate_rows": int(len(candidates)),
        "selected_passed_indices": list(range(take)),
        "target_probabilities": {
            code: float(probabilities[code]) for code in codes
        },
    }
    if take == 0:
        return candidates.head(0).copy(), trace
    if len(candidates) <= take or not codes:
        return candidates.head(take).reset_index(drop=True), trace

    probability_vector = np.asarray(
        [float(probabilities[code]) for code in codes], dtype=float
    )
    if np.any((probability_vector < 0.0) | (probability_vector > 1.0)):
        raise ValueError("Frequency-alignment probabilities must be in [0, 1]")
    phenotype_column = schema["phenotype_column"]
    presence = np.asarray(
        [
            [float(code in set(parse_hpo_codes(value, schema))) for code in codes]
            for value in candidates[phenotype_column]
        ],
        dtype=float,
    )
    selected: list[int] = []
    selected_counts = np.zeros(len(codes), dtype=float)
    available = set(range(len(candidates)))
    final_loss = 0.0
    for step in range(take):
        expected_counts = (step + 1) * probability_vector

        def candidate_key(index: int) -> tuple[float, int]:
            difference = selected_counts + presence[index] - expected_counts
            return (float(np.mean(np.square(difference))), index)

        chosen = min(available, key=candidate_key)
        difference = selected_counts + presence[chosen] - expected_counts
        final_loss = float(np.mean(np.square(difference)))
        selected.append(chosen)
        selected_counts += presence[chosen]
        available.remove(chosen)

    trace.update(
        {
            "applied": True,
            "selected_passed_indices": selected,
            "selection_loss": final_loss,
        }
    )
    return candidates.iloc[selected].reset_index(drop=True), trace


def _integer_interquartile_range(values: list[int]) -> tuple[int, int]:
    if not values:
        return (0, 0)
    q1, q3 = np.quantile(np.asarray(values, dtype=float), [0.25, 0.75])
    return (max(0, int(np.floor(q1))), max(0, int(np.ceil(q3))))


def _largest_numeric_coverage_gap(
    reference: pd.Series,
    accepted: pd.Series,
) -> dict[str, float] | None:
    real = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    synthetic = (
        pd.to_numeric(accepted, errors="coerce").dropna().to_numpy(dtype=float)
    )
    if real.size == 0:
        return None
    unique = np.unique(real)
    if unique.size == 1:
        value = float(unique[0])
        accepted_share = float(np.mean(np.isclose(synthetic, value))) if synthetic.size else 0.0
        gap = max(1.0 - accepted_share, 0.0)
        return {"lower": value, "upper": value, "gap": gap} if gap > 0 else None

    edges = np.unique(np.quantile(real, [0.0, 0.25, 0.5, 0.75, 1.0]))
    if edges.size < 2:
        return None
    real_histogram, _ = np.histogram(real, bins=edges)
    synthetic_histogram, _ = np.histogram(synthetic, bins=edges)
    real_share = real_histogram / real_histogram.sum()
    if synthetic.size:
        synthetic_share = synthetic_histogram / synthetic.size
    else:
        synthetic_share = np.zeros_like(real_share, dtype=float)
    gaps = real_share - synthetic_share
    best = int(np.argmax(gaps))
    if gaps[best] <= 0:
        return None
    return {
        "lower": float(edges[best]),
        "upper": float(edges[best + 1]),
        "gap": float(gaps[best]),
    }


def _coverage_components(
    source_frame: pd.DataFrame,
    accepted: pd.DataFrame,
    schema: dict[str, Any],
    max_examination_items: int,
    numerical_validity: NumericalValidityProfile | None,
) -> dict[str, Any]:
    phenotype_column = schema["phenotype_column"]
    if phenotype_column in source_frame:
        hpo_counts = [
            len(parse_hpo_codes(value, schema))
            for value in source_frame[phenotype_column]
        ]
    else:
        hpo_counts = []

    examination_columns = [
        field["model_column"]
        for field in schema["examination_fields"]
        if field["model_column"] in source_frame
    ]
    missing_counts = (
        source_frame[examination_columns].isna().sum(axis=1).astype(int).tolist()
        if examination_columns
        else []
    )

    examination_targets: list[dict[str, Any]] = []
    missing_gaps: list[tuple[float, str]] = []
    accepted_rows = len(accepted)
    for column in examination_columns:
        target = _largest_numeric_coverage_gap(
            source_frame[column],
            accepted[column] if column in accepted else pd.Series(dtype=float),
        )
        if target is not None:
            target["column"] = column
            examination_targets.append(target)
        real_missing = float(source_frame[column].isna().mean())
        accepted_missing = (
            float(accepted[column].isna().mean())
            if accepted_rows and column in accepted
            else 0.0
        )
        gap = real_missing - accepted_missing
        if gap > 0:
            missing_gaps.append((gap, column))
    examination_targets.sort(
        key=lambda item: (-float(item["gap"]), str(item["column"]))
    )
    examination_targets = examination_targets[:max_examination_items]
    missing_columns = [
        column
        for _, column in sorted(missing_gaps, key=lambda item: (-item[0], item[1]))[
            :max_examination_items
        ]
    ]

    demographic_targets: list[dict[str, Any]] = []
    for field in schema["demographic_fields"]:
        column = field["model_column"]
        if column not in source_frame:
            continue
        accepted_series = (
            accepted[column] if column in accepted else pd.Series(dtype=object)
        )
        if field["value_type"] == "numeric":
            target = _largest_numeric_coverage_gap(
                source_frame[column], accepted_series
            )
            if target is not None:
                demographic_targets.append(
                    {"column": column, "kind": "numeric", **target}
                )
            continue

        real_share = source_frame[column].value_counts(normalize=True, dropna=False)
        accepted_share = accepted_series.value_counts(normalize=True, dropna=False)
        accepted_share_by_value = accepted_share.to_dict()
        candidates: list[tuple[float, str, Any]] = []
        inverse_mapping = {
            mapped: raw for raw, mapped in field.get("mapping", {}).items()
        }
        for value, share in real_share.items():
            gap = float(share) - float(accepted_share_by_value.get(value, 0.0))
            if gap > 0:
                rendered = inverse_mapping.get(value, value)
                candidates.append((gap, str(rendered), rendered))
        if candidates:
            gap, _, rendered = min(
                candidates, key=lambda item: (-item[0], item[1])
            )
            demographic_targets.append(
                {
                    "column": column,
                    "kind": "categorical",
                    "value": rendered,
                    "gap": gap,
                }
            )

    return {
        "coverage_available": True,
        "hpo_count_range": _integer_interquartile_range(hpo_counts),
        "examination_targets": examination_targets,
        "missing_count_range": _integer_interquartile_range(missing_counts),
        "missing_columns": missing_columns,
        "demographic_targets": demographic_targets,
        "numerical_bounds": (
            dict(numerical_validity.bounds) if numerical_validity else {}
        ),
    }


def _select_shared_hpo_anchors(
    train_df: pd.DataFrame,
    disease_train: pd.DataFrame,
    schema: dict[str, Any],
    minimum_joint_support: int,
    max_anchors: int,
) -> tuple[list[str], int | None]:
    """Select a fixed disease-relevant HPO subset shared by enough patients."""
    candidate_counts = _hpo_counts(disease_train, schema)
    candidates = sorted(candidate_counts)
    if not candidates or max_anchors <= 0:
        return [], None

    candidate_set = set(candidates)
    masks = {code: 0 for code in candidates}
    phenotype_column = schema["phenotype_column"]
    for position, value in enumerate(train_df[phenotype_column]):
        for code in set(parse_hpo_codes(value, schema)) & candidate_set:
            masks[code] |= 1 << position
    marginal_support = {
        code: masks[code].bit_count()
        for code in candidates
    }

    for size in range(min(max_anchors, len(candidates)), 0, -1):
        valid: list[tuple[int, int, tuple[str, ...]]] = []
        for subset in combinations(candidates, size):
            shared_mask = masks[subset[0]]
            for code in subset[1:]:
                shared_mask &= masks[code]
            joint_support = shared_mask.bit_count()
            if joint_support >= minimum_joint_support:
                valid.append(
                    (
                        joint_support,
                        sum(marginal_support[code] for code in subset),
                        subset,
                    )
                )
        if valid:
            joint_support, _, chosen = min(valid)
            return list(chosen), joint_support
    return [], None


def build_phenotype_profile(
    train_df: pd.DataFrame,
    disease_train: pd.DataFrame,
    accepted_disease: pd.DataFrame,
    schema: dict[str, Any],
    target_label: int,
    minimum_cohort_rows: int,
    max_items: int,
    minimum_support: int = 2,
    max_examination_items: int = 4,
    numerical_validity: NumericalValidityProfile | None = None,
    max_shared_anchors: int = 4,
) -> PhenotypeProfile:
    """Build one privacy-filtered, accepted-aware clinical coverage profile."""
    if minimum_cohort_rows <= 0:
        raise ValueError("minimum_cohort_rows must be positive")
    if max_items <= 0:
        raise ValueError("max_items must be positive")
    if minimum_support < 2:
        raise ValueError("minimum_support must be at least 2")
    if max_shared_anchors <= 0:
        raise ValueError("max_shared_anchors must be positive")
    if max_examination_items <= 0:
        raise ValueError("max_examination_items must be positive")

    label_column = schema["label_column"]
    table_train = train_df[
        train_df[label_column].astype(int) == int(target_label)
    ]
    if len(disease_train) >= minimum_cohort_rows:
        source = "disease"
        source_frame = disease_train
        coverage_source = "disease"
        coverage_frame = disease_train
    elif len(table_train) >= minimum_cohort_rows:
        source = "table"
        source_frame = table_train
        coverage_source = "table"
        coverage_frame = table_train
    else:
        anchors, joint_support = _select_shared_hpo_anchors(
            train_df,
            disease_train,
            schema,
            minimum_joint_support=minimum_cohort_rows,
            max_anchors=min(max_shared_anchors, max_items),
        )
        disease_counts = _hpo_counts(disease_train, schema)
        accepted_counts = _hpo_counts(accepted_disease, schema)
        disease_rows = len(disease_train)
        accepted_rows = len(accepted_disease)

        def anchor_rank(code: str) -> tuple[float, str]:
            real_prevalence = disease_counts[code] / disease_rows
            accepted_prevalence = (
                accepted_counts[code] / accepted_rows
                if accepted_rows
                else 0.0
            )
            return (
                -max(real_prevalence - accepted_prevalence, 0.0),
                code,
            )

        return PhenotypeProfile(
            codes=sorted(anchors, key=anchor_rank),
            source="shared_anchor" if anchors else "knowledge_only",
            source_rows=len(train_df),
            minimum_support=minimum_cohort_rows,
            coverage_available=False,
            coverage_source="unavailable_below_support",
            hpo_count_range=(0, 0),
            examination_targets=[],
            missing_count_range=(0, 0),
            missing_columns=[],
            demographic_targets=[],
            numerical_bounds=(
                dict(numerical_validity.bounds) if numerical_validity else {}
            ),
            joint_support=joint_support,
        )

    source_counts = _hpo_counts(source_frame, schema)
    source_rows = len(source_frame)
    accepted_counts = _hpo_counts(accepted_disease, schema)
    accepted_rows = len(accepted_disease)
    candidates = [
        code
        for code, support in source_counts.items()
        if support >= minimum_support
    ]

    def rank(code: str) -> tuple[float, float, str]:
        real_prevalence = source_counts[code] / source_rows
        accepted_prevalence = (
            accepted_counts[code] / accepted_rows if accepted_rows else 0.0
        )
        coverage_gap = max(real_prevalence - accepted_prevalence, 0.0)
        return (-coverage_gap, -real_prevalence, code)

    components = _coverage_components(
        coverage_frame,
        accepted_disease,
        schema,
        max_examination_items,
        numerical_validity,
    )
    return PhenotypeProfile(
        codes=sorted(candidates, key=rank)[:max_items],
        source=source,
        source_rows=source_rows,
        minimum_support=minimum_support,
        coverage_source=coverage_source,
        **components,
    )


def disease_to_label_mapping(
    dataframe: pd.DataFrame,
    schema: dict[str, Any],
) -> dict[str, list[int]]:
    """Return every observed Table label for each non-empty disease name."""
    disease_column = schema.get("disease_name_column")
    label_column = schema["label_column"]
    if not disease_column or disease_column not in dataframe:
        raise ValueError("The EHR disease-name column is unavailable")

    observed: dict[str, set[int]] = {}
    for _, row in dataframe[[disease_column, label_column]].iterrows():
        label = int(row[label_column])
        for name in parse_disease_names([row[disease_column]]):
            observed.setdefault(name, set()).add(label)
    return {name: sorted(labels) for name, labels in sorted(observed.items())}


def rows_for_disease(
    dataframe: pd.DataFrame,
    schema: dict[str, Any],
    disease_name: str,
    target_label: int | None = None,
) -> pd.DataFrame:
    disease_column = schema["disease_name_column"]
    mask = dataframe[disease_column].map(
        lambda value: disease_name in parse_disease_names([value])
    ).astype(bool)
    if target_label is not None:
        mask &= dataframe[schema["label_column"]].astype(int) == int(target_label)
    return dataframe.loc[mask].copy()


def resolve_disease_weights(
    train_df: pd.DataFrame,
    schema: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[dict[tuple[str, int], float], dict[str, list[int]]]:
    """Resolve weights over observed disease-name/Table pairs."""
    mapping = disease_to_label_mapping(train_df, schema)
    if not mapping:
        raise ValueError("D_train contains no mapped IUIS disease names")

    pair_counts: Counter[tuple[str, int]] = Counter()
    disease_column = schema["disease_name_column"]
    label_column = schema["label_column"]
    for _, row in train_df[[disease_column, label_column]].iterrows():
        label = int(row[label_column])
        pair_counts.update((name, label) for name in parse_disease_names([row[disease_column]]))
    name_counts = Counter()
    for (name, _), count in pair_counts.items():
        name_counts[name] += count
    mode = settings.get("mode", "empirical")
    if mode == "empirical":
        name_weights = {name: float(name_counts[name]) for name in mapping}
    elif mode == "uniform":
        name_weights = {name: 1.0 for name in mapping}
    elif mode == "custom":
        custom = settings.get("custom")
        if not isinstance(custom, dict):
            raise ValueError("custom disease weights require a mapping")
        name_weights = {str(name): float(weight) for name, weight in custom.items()}
        if set(name_weights) != set(mapping):
            raise ValueError(
                "custom disease weights must cover exactly the diseases in D_train"
            )
    else:
        raise ValueError(
            "disease_weights.mode must be empirical, uniform, or custom"
        )
    if any(not np.isfinite(weight) or weight < 0 for weight in name_weights.values()):
        raise ValueError("Disease weights must be finite and non-negative")
    name_total = float(sum(name_weights.values()))
    if name_total <= 0:
        raise ValueError("At least one disease weight must be positive")
    weights = {
        pair: (name_weights[pair[0]] / name_total) * (count / name_counts[pair[0]])
        for pair, count in sorted(pair_counts.items())
    }
    return weights, mapping


def build_disease_schedule(
    train_df: pd.DataFrame,
    schema: dict[str, Any],
    settings: dict[str, Any],
    n_samples: int,
    seed: int,
) -> DiseaseSchedule:
    if n_samples <= 0:
        raise ValueError("generation.n_samples must be positive")
    weights, mapping = resolve_disease_weights(train_df, schema, settings)
    targets = sorted(weights)
    sampled_indices = np.random.default_rng(seed).choice(
        len(targets),
        size=n_samples,
        replace=True,
        p=[weights[target] for target in targets],
    )
    schedule = [targets[int(index)] for index in sampled_indices]
    counts = Counter(schedule)
    return DiseaseSchedule(
        targets=schedule,
        weights=weights,
        target_counts={target: int(counts[target]) for target in targets},
        disease_to_labels=mapping,
    )


def select_random_exemplars(
    train_df: pd.DataFrame,
    schema: dict[str, Any],
    target_label: int,
    max_exemplars: int,
    rng: np.random.Generator,
) -> ExemplarSelection:
    if max_exemplars <= 0:
        raise ValueError("retrieval.max_exemplars must be positive")
    label_column = schema["label_column"]
    candidates = train_df[train_df[label_column].astype(int) == int(target_label)]
    if candidates.empty:
        raise ValueError(f"No real exemplars for disease label {target_label}")
    count = min(max_exemplars, len(candidates))
    positions = rng.choice(len(candidates), size=count, replace=False)
    selected = candidates.iloc[positions].reset_index(drop=True)
    record_id_column = schema.get("record_id_column")
    selected_ids = (
        selected[record_id_column].astype(str).tolist()
        if record_id_column in selected.columns
        else []
    )
    return ExemplarSelection(
        dataframe=selected,
        trace={
            "mode": "same_subtype_random",
            "target_subtype_id": int(target_label),
            "candidate_count": int(len(candidates)),
            "selected_count": int(len(selected)),
            "selected_record_ids": selected_ids,
            "fallback_reason": "random_exemplar_selection",
            "distance": None,
        },
    )


def build_residual_frame(
    dataframe: pd.DataFrame,
    schema: dict[str, Any],
) -> ResidualFrame:
    """Expand HPO only for TabGen-compatible residual distance calculations."""
    phenotype_column = schema["phenotype_column"]
    missing = [
        column
        for column in schema["feature_columns"]
        if column not in dataframe.columns
    ]
    if missing:
        raise ValueError(f"Residual input is missing feature columns: {missing}")

    hpo_sets = [
        set(parse_hpo_codes(value, schema))
        for value in dataframe[phenotype_column]
    ]
    values: dict[str, Any] = {}
    categorical: set[str] = set()
    for code in schema["hpo_vocabulary"]:
        column = f"__hpo__{code.replace(':', '_')}"
        values[column] = [int(code in present) for present in hpo_sets]
        categorical.add(column)

    categorical_features = set(schema["categorical_columns"])
    for column in schema["feature_columns"]:
        if column == phenotype_column:
            continue
        if column in categorical_features:
            values[column] = dataframe[column].astype(object).tolist()
            categorical.add(column)
        else:
            values[column] = pd.to_numeric(dataframe[column], errors="coerce").tolist()
    return ResidualFrame(
        dataframe=pd.DataFrame(values, index=dataframe.index),
        categorical_columns=categorical,
    )


def jensen_shannon_distance(
    ground_truth: np.ndarray,
    synthetic: np.ndarray,
    normalize: bool = True,
    n_bins: int = 10,
    laplace_smoothing: bool = True,
) -> float:
    """Minimal port of TabGen-ICL's column-averaged JSD implementation."""
    if ground_truth.shape[1] != synthetic.shape[1]:
        raise ValueError("JSD inputs must have the same number of columns")
    distances: list[float] = []
    for column_index in range(ground_truth.shape[1]):
        real_column = ground_truth[:, column_index]
        synthetic_column = synthetic[:, column_index]
        local_bins = min(n_bins, len(np.unique(real_column)))
        real_histogram, bin_edges = np.histogram(
            real_column,
            bins=local_bins,
            density=normalize,
        )
        synthetic_counts, _ = np.histogram(
            synthetic_column,
            bins=bin_edges,
            density=False,
        )
        if synthetic_counts.sum() == 0:
            # TabGen derives its bins from the first input. With disjoint
            # numerical support, every value in the second input can fall
            # outside those bins and density=True then divides 0 by 0. Only
            # for that undefined edge case, rebuild bins on the joint support.
            joint_values = np.concatenate([real_column, synthetic_column])
            local_bins = min(n_bins, len(np.unique(joint_values)))
            bin_edges = np.histogram_bin_edges(
                joint_values,
                bins=local_bins,
            )
            real_histogram, _ = np.histogram(
                real_column,
                bins=bin_edges,
                density=normalize,
            )
        synthetic_histogram, _ = np.histogram(
            synthetic_column,
            bins=bin_edges,
            density=normalize,
        )
        if laplace_smoothing:
            real_histogram += 1
            synthetic_histogram += 1
        real_histogram = real_histogram / np.sum(real_histogram)
        synthetic_histogram = synthetic_histogram / np.sum(synthetic_histogram)
        with np.errstate(invalid="ignore"):
            value = float(
                distance.jensenshannon(real_histogram, synthetic_histogram)
            )
        if np.isnan(value):
            midpoint = (real_histogram + synthetic_histogram) / 2.0
            divergence = 0.5 * float(
                np.sum(special.rel_entr(real_histogram, midpoint))
                + np.sum(special.rel_entr(synthetic_histogram, midpoint))
            )
            tolerance = 64.0 * np.finfo(float).eps
            if (
                np.isfinite(divergence)
                and -tolerance <= divergence <= 0.0
            ):
                value = 0.0
            else:
                raise RuntimeError(
                    f"NaN in JSD calculation for column {column_index}"
                )
        distances.append(value)
    return float(np.mean(distances))


def _tabgen_encode(
    ground_truth: pd.DataFrame,
    comparison: pd.DataFrame,
    categorical_columns: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    if list(ground_truth.columns) != list(comparison.columns):
        raise ValueError("Residual distance inputs must have identical columns")
    if ground_truth.empty or comparison.empty:
        raise ValueError("Residual distance inputs cannot be empty")
    if ground_truth.isna().any().any() or comparison.isna().any().any():
        raise ValueError(
            "Residual distance encountered missing values; the numerical "
            "missing-value policy must be frozen before EHR residual retrieval"
        )

    encoded_real: list[np.ndarray] = []
    encoded_comparison: list[np.ndarray] = []
    for column in ground_truth.columns:
        real_values = ground_truth[column].to_numpy().reshape(-1, 1)
        comparison_values = comparison[column].to_numpy().reshape(-1, 1)
        if column in categorical_columns:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoded_real.append(encoder.fit_transform(real_values))
            encoded_comparison.append(encoder.transform(comparison_values))
        else:
            encoded_real.append(real_values.astype(float))
            encoded_comparison.append(comparison_values.astype(float))
    return (
        np.concatenate(encoded_real, axis=1),
        np.concatenate(encoded_comparison, axis=1),
    )


def tabgen_distance(
    ground_truth: pd.DataFrame,
    comparison: pd.DataFrame,
    categorical_columns: set[str],
    metric: str,
) -> float:
    """Compute JSD or K-S distance with TabGen preprocessing."""
    encoded_real, encoded_comparison = _tabgen_encode(
        ground_truth,
        comparison,
        categorical_columns,
    )
    encoded_comparison = resample(
        encoded_comparison,
        replace=True,
        n_samples=encoded_real.shape[0],
        random_state=2024,
    )
    if metric == "jsd":
        return jensen_shannon_distance(encoded_comparison, encoded_real)
    if metric == "ksd":
        statistics = [
            stats.ks_2samp(
                encoded_comparison[:, column],
                encoded_real[:, column],
                method="asymp",
            ).statistic
            for column in range(encoded_real.shape[1])
        ]
        return float(np.mean(statistics))
    raise ValueError("Residual metric must be jsd or ksd")


def build_candidate_groups(
    residual_frame: ResidualFrame,
    selected_column: str,
    max_exemplars: int,
    numerical_bins: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    if max_exemplars <= 0:
        raise ValueError("retrieval.max_exemplars must be positive")
    if numerical_bins <= 0:
        raise ValueError("retrieval.numerical_bins must be positive")
    if selected_column not in residual_frame.dataframe.columns:
        raise ValueError(f"Unknown residual grouping column: {selected_column}")
    values = residual_frame.dataframe[selected_column]
    if values.isna().any():
        raise ValueError(
            f"Residual grouping column {selected_column!r} contains missing values"
        )

    raw_groups: list[np.ndarray] = []
    if selected_column in residual_frame.categorical_columns:
        for value in pd.unique(values):
            raw_groups.append(np.flatnonzero(values.to_numpy() == value))
    else:
        numeric = values.to_numpy(dtype=float)
        if np.min(numeric) == np.max(numeric):
            raw_groups.append(np.arange(len(numeric)))
        else:
            edges = np.linspace(
                float(np.min(numeric)),
                float(np.max(numeric)),
                num=numerical_bins + 1,
            )
            assignments = np.digitize(numeric, edges[1:-1], right=False)
            for bin_id in range(numerical_bins):
                positions = np.flatnonzero(assignments == bin_id)
                if len(positions):
                    raw_groups.append(positions)

    groups: list[np.ndarray] = []
    for positions in raw_groups:
        if not len(positions):
            continue
        if len(positions) > max_exemplars:
            positions = np.asarray(
                rng.choice(positions, size=max_exemplars, replace=False),
                dtype=int,
            )
        groups.append(np.asarray(positions, dtype=int))
    if not groups:
        raise ValueError("Residual grouping produced no non-empty candidate groups")
    return groups


def select_residual_exemplars(
    train_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    schema: dict[str, Any],
    target_label: int,
    max_exemplars: int,
    numerical_bins: int,
    retrieval_round: int,
    rng: np.random.Generator,
    numerical_missing_policy: str | None = None,
) -> ExemplarSelection:
    label_column = schema["label_column"]
    real = train_df[train_df[label_column].astype(int) == int(target_label)].copy()
    synthetic = synthetic_df[
        synthetic_df[label_column].astype(int) == int(target_label)
    ].copy()
    if real.empty:
        raise ValueError(f"No real records for disease label {target_label}")
    if synthetic.empty:
        raise ValueError("Residual retrieval requires prior same-disease synthetic rows")

    real_residual = build_residual_frame(real, schema)
    synthetic_residual = build_residual_frame(synthetic, schema)
    if real_residual.categorical_columns != synthetic_residual.categorical_columns:
        raise RuntimeError("Real and synthetic residual categorical schemas differ")
    categorical_columns = real_residual.categorical_columns
    numerical_columns = [
        column
        for column in real_residual.dataframe.columns
        if column not in categorical_columns
    ]
    categorical_missing = [
        column
        for column in categorical_columns
        if real_residual.dataframe[column].isna().any()
        or synthetic_residual.dataframe[column].isna().any()
    ]
    if categorical_missing:
        raise ValueError(
            "Residual categorical fields contain missing values: "
            f"{categorical_missing}"
        )
    ignored_all_missing_numerical_columns: list[str] = []
    numerical_missing = [
        column
        for column in numerical_columns
        if real_residual.dataframe[column].isna().any()
        or synthetic_residual.dataframe[column].isna().any()
    ]
    if numerical_missing:
        if numerical_missing_policy == "error":
            raise ValueError(
                "Residual numerical fields contain missing values under the "
                f"error policy: {numerical_missing}"
            )
        if numerical_missing_policy != "median":
            raise ValueError(
                "retrieval.numerical_missing_policy must be error or median "
                "when residual inputs contain missing values"
            )
        medians = real_residual.dataframe[numerical_columns].median(axis=0)
        ignored_all_missing_numerical_columns = (
            medians[medians.isna()].index.tolist()
        )
        if ignored_all_missing_numerical_columns:
            real_residual.dataframe = real_residual.dataframe.drop(
                columns=ignored_all_missing_numerical_columns
            )
            synthetic_residual.dataframe = synthetic_residual.dataframe.drop(
                columns=ignored_all_missing_numerical_columns
            )
            numerical_columns = [
                column
                for column in numerical_columns
                if column not in ignored_all_missing_numerical_columns
            ]
            medians = medians.drop(
                index=ignored_all_missing_numerical_columns
            )
        if numerical_columns:
            real_residual.dataframe[numerical_columns] = real_residual.dataframe[
                numerical_columns
            ].fillna(medians)
            synthetic_residual.dataframe[numerical_columns] = (
                synthetic_residual.dataframe[numerical_columns].fillna(medians)
            )
    selected_column = str(rng.choice(real_residual.dataframe.columns.to_numpy()))
    groups = build_candidate_groups(
        real_residual,
        selected_column,
        min(max_exemplars, len(real)),
        numerical_bins,
        rng,
    )
    metric = "jsd" if retrieval_round % 2 == 0 else "ksd"
    distances: list[float] = []
    for positions in groups:
        candidate = real_residual.dataframe.iloc[positions]
        comparison = pd.concat(
            [synthetic_residual.dataframe, candidate],
            ignore_index=True,
        )
        distances.append(
            tabgen_distance(
                real_residual.dataframe,
                comparison,
                real_residual.categorical_columns,
                metric,
            )
        )
    best_index = int(np.argmin(distances))
    best_positions = groups[best_index]
    selected = real.iloc[best_positions].reset_index(drop=True)
    record_id_column = schema.get("record_id_column")
    selected_ids = (
        selected[record_id_column].astype(str).tolist()
        if record_id_column in selected.columns
        else []
    )
    return ExemplarSelection(
        dataframe=selected,
        trace={
            "mode": "same_subtype_residual",
            "target_subtype_id": int(target_label),
            "candidate_count": int(len(real)),
            "selected_count": int(len(selected)),
            "selected_record_ids": selected_ids,
            "fallback_reason": None,
            "retrieval_round": int(retrieval_round),
            "distance": metric,
            "selected_column": selected_column,
            "candidate_group_sizes": [int(len(group)) for group in groups],
            "candidate_distances": [float(value) for value in distances],
            "selected_group_index": best_index,
            "numerical_missing_policy": numerical_missing_policy,
            "ignored_all_missing_numerical_columns": (
                ignored_all_missing_numerical_columns
            ),
        },
    )
