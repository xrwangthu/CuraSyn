"""CLLM data curation based on classifier learning dynamics.

The implementation preserves the reference CLLM definitions and checkpoint
traversal while using NumPy rather than Torch for probability bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.data import parse_hpo_codes


class Curator:
    """Track gold-label probabilities over classifier checkpoints."""

    def __init__(self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
        self.x = x
        self.y = np.asarray(y, dtype=int).reshape(-1)
        self._gold_label_probabilities: np.ndarray | None = None
        self._true_probabilities: np.ndarray | None = None
        self.curation_metric: str | None = None
        self.curation_xthreshold: float | None = None

    def on_iteration_end(self, clf: XGBClassifier, iteration: int) -> None:
        probabilities = np.asarray(
            clf.predict_proba(self.x, iteration_range=(0, iteration)),
            dtype=float,
        )
        if probabilities.ndim != 2:
            raise ValueError(f"Expected a probability matrix, got shape {probabilities.shape}")
        if len(probabilities) != len(self.y):
            raise ValueError("Probability and label row counts do not match")
        if np.any(self.y < 0) or np.any(self.y >= probabilities.shape[1]):
            raise ValueError("Curation labels must index classifier probability columns")

        row_ids = np.arange(len(self.y))
        gold = probabilities[row_ids, self.y]
        predicted = probabilities.max(axis=1)
        gold = np.expand_dims(gold, axis=-1)
        predicted = np.expand_dims(predicted, axis=-1)

        if self._gold_label_probabilities is None:
            self._gold_label_probabilities = gold
            self._true_probabilities = predicted
        else:
            self._gold_label_probabilities = np.hstack(
                [self._gold_label_probabilities, gold]
            )
            self._true_probabilities = np.hstack(
                [self._true_probabilities, predicted]
            )

    @property
    def gold_label_probabilities(self) -> np.ndarray:
        if self._gold_label_probabilities is None:
            raise RuntimeError("No learning-dynamics checkpoints have been recorded")
        return self._gold_label_probabilities

    @property
    def true_probabilities(self) -> np.ndarray:
        if self._true_probabilities is None:
            raise RuntimeError("No learning-dynamics checkpoints have been recorded")
        return self._true_probabilities

    @property
    def confidence(self) -> np.ndarray:
        return np.mean(self.gold_label_probabilities, axis=-1)

    @property
    def aleatoric(self) -> np.ndarray:
        probabilities = self.gold_label_probabilities
        return np.mean(probabilities * (1 - probabilities), axis=-1)

    @property
    def variability(self) -> np.ndarray:
        return np.std(self.gold_label_probabilities, axis=-1)

    @property
    def correctness(self) -> np.ndarray:
        return np.mean(self.gold_label_probabilities > 0.5, axis=-1)

    @property
    def entropy(self) -> np.ndarray:
        probabilities = self.gold_label_probabilities
        return -np.sum(probabilities * np.log(probabilities + 1e-12), axis=-1)

    @property
    def mi(self) -> np.ndarray:
        probabilities = self.gold_label_probabilities
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=-1)
        mean_probability = np.mean(probabilities, axis=1)
        entropy_expected = -np.sum(
            mean_probability * np.log(mean_probability + 1e-12),
            axis=-1,
        )
        return entropy - entropy_expected


def get_groups(
    confidence: np.ndarray,
    uncertainty: np.ndarray,
    curation_xthreshold: float,
    curation_ythreshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the CLLM easy/ambiguous/hard partition."""
    confidence_low = curation_ythreshold
    confidence_high = 1 - curation_ythreshold

    hard = np.where(
        (confidence <= confidence_low) & (uncertainty <= curation_xthreshold)
    )[0]
    easy = np.where(
        (confidence >= confidence_high) & (uncertainty <= curation_xthreshold)
    )[0]
    assigned = set(np.concatenate([hard, easy]).tolist())
    ambiguous = np.asarray(
        [idx for idx in range(len(confidence)) if idx not in assigned],
        dtype=int,
    )
    return easy, ambiguous, hard


def data_centric_curation(
    x_train_orig: pd.DataFrame | np.ndarray,
    y_train_orig: pd.Series | np.ndarray,
    x_check: pd.DataFrame | np.ndarray,
    y_check: pd.Series | np.ndarray,
    curation_metric: str = "aleatoric",
    retrain: bool = False,
    nest: int = 100,
    curation_ythreshold: float = 0.2,
    curation_xthreshold: float = 0.2,
    n_jobs: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Curator]:
    """Run the reference CLLM learning-dynamics curation algorithm."""
    if nest < 2:
        raise ValueError("curation.n_estimators must be at least 2")
    if not 0 <= curation_ythreshold <= 0.5:
        raise ValueError("curation.confidence_threshold must be in [0, 0.5]")

    clf = XGBClassifier(n_estimators=nest, n_jobs=n_jobs)
    clf.fit(x_train_orig, y_train_orig)
    if retrain:
        clf = XGBClassifier(n_estimators=nest, n_jobs=n_jobs)
        clf.fit(x_check, y_check)

    curator = Curator(x_check, y_check)
    for iteration in range(1, nest):
        curator.on_iteration_end(clf, iteration)

    metrics = {
        "aleatoric": curator.aleatoric,
        "epistemic": curator.variability,
        "entropy": curator.entropy,
        "mi": curator.mi,
    }
    if curation_metric not in metrics:
        supported = ", ".join(metrics)
        raise ValueError(
            f"Unsupported curation metric {curation_metric!r}; supported: {supported}"
        )
    uncertainty = metrics[curation_metric]
    resolved_xthreshold = float(curation_xthreshold)
    if resolved_xthreshold == 0:
        resolved_xthreshold = 0.75 * float(
            np.max(uncertainty) - np.min(uncertainty)
        )

    easy, ambiguous, hard = get_groups(
        confidence=curator.confidence,
        uncertainty=uncertainty,
        curation_xthreshold=resolved_xthreshold,
        curation_ythreshold=float(curation_ythreshold),
    )
    curator.curation_metric = curation_metric
    curator.curation_xthreshold = resolved_xthreshold
    return easy, ambiguous, hard, curator


def curate_generated(
    train_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    schema: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Curate generated rows and return serializable dynamics and indices."""
    label = schema["label_column"]
    curation = config["curation"]
    easy, ambiguous, hard, curator = data_centric_curation(
        x_train_orig=train_df.drop(columns=[label]),
        y_train_orig=train_df[label],
        x_check=generated_df.drop(columns=[label]),
        y_check=generated_df[label],
        curation_metric=curation.get("metric", "aleatoric"),
        retrain=bool(curation.get("retrain", False)),
        nest=int(curation["n_estimators"]),
        curation_ythreshold=float(curation["confidence_threshold"]),
        curation_xthreshold=float(curation["uncertainty_threshold"]),
        n_jobs=(int(curation["n_jobs"]) if curation.get("n_jobs") is not None else None),
    )
    metric_name = str(curator.curation_metric)
    metric_values = {
        "aleatoric": curator.aleatoric,
        "epistemic": curator.variability,
        "entropy": curator.entropy,
        "mi": curator.mi,
    }[metric_name]
    selected = np.concatenate([easy, ambiguous]).astype(int)
    return {
        "algorithm": "cllm_learning_dynamics",
        "checkpoints": int(curator.gold_label_probabilities.shape[1]),
        "metric": metric_name,
        "thresholds": {
            "confidence": float(curation["confidence_threshold"]),
            "uncertainty_configured": float(curation["uncertainty_threshold"]),
            "uncertainty_resolved": float(curator.curation_xthreshold),
        },
        "easy": easy.astype(int).tolist(),
        "ambiguous": ambiguous.astype(int).tolist(),
        "hard": hard.astype(int).tolist(),
        "selected": selected.tolist(),
        "confidence": curator.confidence.tolist(),
        "uncertainty": metric_values.tolist(),
        "aleatoric": curator.aleatoric.tolist(),
        "epistemic": curator.variability.tolist(),
        "correctness": curator.correctness.tolist(),
        "gold_label_probabilities": curator.gold_label_probabilities.tolist(),
    }


PID_CURATION_EPSILON = 1.0e-8


@dataclass
class PidFeatureEncoder:
    """Train-fitted classifier encoding for semantic PID table fields."""

    schema: dict[str, Any]
    config: dict[str, Any]

    def __post_init__(self) -> None:
        if self.config.get("hpo_encoding") != "multihot":
            raise ValueError("curation.encoder.hpo_encoding must be 'multihot'")
        if self.config.get("categorical_encoding") != "onehot":
            raise ValueError(
                "curation.encoder.categorical_encoding must be 'onehot'"
            )
        missing_policy = self.config.get("numerical_missing_policy")
        if missing_policy not in {"error", "median"}:
            raise ValueError(
                "curation.encoder.numerical_missing_policy must be error or median"
            )
        self.numerical_columns = list(self.schema["numerical_columns"])
        self.categorical_columns = [
            column
            for column in self.schema["categorical_columns"]
            if column != self.schema["phenotype_column"]
        ]
        self.hpo_vocabulary = list(self.schema["hpo_vocabulary"])
        self.active_numerical_columns: list[str] = []
        self.skipped_numerical_columns: list[str] = []
        self.numerical_fill_values: dict[str, float] = {}
        self.categorical_values: dict[str, list[Any]] = {}
        self.feature_names: list[str] = []
        self.fitted = False

    def fit(self, train_df: pd.DataFrame) -> "PidFeatureEncoder":
        self.active_numerical_columns = []
        self.skipped_numerical_columns = []
        self.numerical_fill_values = {}
        for column in self.numerical_columns:
            values = pd.to_numeric(train_df[column], errors="coerce")
            if values.notna().sum() == 0:
                if self.config["numerical_missing_policy"] == "error":
                    raise ValueError(
                        f"Classifier encoder cannot fit all-missing column {column!r}"
                    )
                self.skipped_numerical_columns.append(column)
                continue
            if (
                self.config["numerical_missing_policy"] == "error"
                and values.isna().any()
            ):
                raise ValueError(
                    f"Classifier encoder found missing values in {column!r} "
                    "under the error policy"
                )
            self.active_numerical_columns.append(column)
            self.numerical_fill_values[column] = float(values.median())

        field_index = {
            field["model_column"]: field
            for field in [
                *self.schema["demographic_fields"],
                *self.schema["examination_fields"],
            ]
        }
        for column in self.categorical_columns:
            field = field_index[column]
            mapping = field.get("mapping")
            if mapping:
                categories = list(mapping.values())
            else:
                categories = train_df[column].dropna().drop_duplicates().tolist()
            if not categories:
                raise ValueError(
                    f"Classifier encoder found no categories for {column!r}"
                )
            self.categorical_values[column] = categories

        self.feature_names = [
            *[f"hpo::{code}" for code in self.hpo_vocabulary],
            *[
                f"{column}::{category}"
                for column in self.categorical_columns
                for category in self.categorical_values[column]
            ],
            *self.active_numerical_columns,
        ]
        self.fitted = True
        return self

    def transform(self, dataframe: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("PidFeatureEncoder must be fitted before transform")
        blocks: list[np.ndarray] = []
        hpo_index = {code: index for index, code in enumerate(self.hpo_vocabulary)}
        hpo = np.zeros((len(dataframe), len(self.hpo_vocabulary)), dtype=float)
        for row_index, value in enumerate(
            dataframe[self.schema["phenotype_column"]]
        ):
            for code in parse_hpo_codes(value, self.schema):
                hpo[row_index, hpo_index[code]] = 1.0
        blocks.append(hpo)

        for column in self.categorical_columns:
            categories = self.categorical_values[column]
            values = dataframe[column]
            unknown = {
                value
                for value in values.dropna().unique().tolist()
                if value not in categories
            }
            if unknown:
                raise ValueError(
                    f"Classifier encoder found unknown values in {column!r}: "
                    f"{sorted(unknown, key=str)}"
                )
            encoded = np.column_stack(
                [
                    (values == category).to_numpy(dtype=float)
                    for category in categories
                ]
            )
            blocks.append(encoded)

        numerical = dataframe[self.active_numerical_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        if (
            self.config["numerical_missing_policy"] == "error"
            and numerical.isna().any().any()
        ):
            missing = numerical.columns[numerical.isna().any()].tolist()
            raise ValueError(
                "Classifier encoder found missing values under the error policy: "
                f"{missing}"
            )
        numerical = numerical.fillna(self.numerical_fill_values)
        blocks.append(numerical.to_numpy(dtype=float))
        return np.hstack(blocks)

    def fit_transform(self, train_df: pd.DataFrame) -> np.ndarray:
        return self.fit(train_df).transform(train_df)

    def metadata(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("PidFeatureEncoder has not been fitted")
        return {
            "hpo_encoding": self.config["hpo_encoding"],
            "categorical_encoding": self.config["categorical_encoding"],
            "numerical_missing_policy": self.config["numerical_missing_policy"],
            "n_features": len(self.feature_names),
            "active_numerical_columns": list(self.active_numerical_columns),
            "skipped_numerical_columns": list(self.skipped_numerical_columns),
            "numerical_fill_values": dict(self.numerical_fill_values),
            "categorical_values": dict(self.categorical_values),
        }


def subtype_calibrated_scores(
    real_labels: np.ndarray,
    real_confidence: np.ndarray,
    real_uncertainty: np.ndarray,
    generated_labels: np.ndarray,
    generated_confidence: np.ndarray,
    generated_uncertainty: np.ndarray,
    confidence_threshold: float,
    uncertainty_threshold: float,
    epsilon: float = PID_CURATION_EPSILON,
) -> dict[str, Any]:
    """Standardize subtype scores and apply the rejection rule."""
    if float(epsilon) != PID_CURATION_EPSILON:
        raise ValueError("PID curation epsilon is fixed at 1e-8 by the method")
    if not np.isfinite(
        [float(confidence_threshold), float(uncertainty_threshold)]
    ).all():
        raise ValueError("PID curation thresholds must be finite")
    real_labels = np.asarray(real_labels, dtype=int).reshape(-1)
    real_confidence = np.asarray(real_confidence, dtype=float).reshape(-1)
    real_uncertainty = np.asarray(real_uncertainty, dtype=float).reshape(-1)
    generated_labels = np.asarray(generated_labels, dtype=int).reshape(-1)
    generated_confidence = np.asarray(generated_confidence, dtype=float).reshape(-1)
    generated_uncertainty = np.asarray(generated_uncertainty, dtype=float).reshape(-1)
    if not (
        len(real_labels) == len(real_confidence) == len(real_uncertainty)
        and len(generated_labels)
        == len(generated_confidence)
        == len(generated_uncertainty)
    ):
        raise ValueError("PID curation score inputs have inconsistent lengths")
    arrays = [
        real_confidence,
        real_uncertainty,
        generated_confidence,
        generated_uncertainty,
    ]
    if any(not np.isfinite(values).all() for values in arrays):
        raise ValueError("PID curation scores must be finite")

    pooled_confidence_std = float(np.std(real_confidence, ddof=0))
    pooled_uncertainty_std = float(np.std(real_uncertainty, ddof=0))

    def calibration_scale(
        subtype_count: int,
        subtype_std: float,
        pooled_std: float,
        metric: str,
    ) -> tuple[float, str]:
        if subtype_count >= 2 and subtype_std > epsilon:
            return subtype_std, "subtype"
        if pooled_std <= epsilon:
            raise ValueError(
                "PID curation cannot calibrate "
                f"{metric}: both subtype and pooled real-data scales are "
                "zero or numerically negligible"
            )
        return pooled_std, "pooled"

    references: dict[str, dict[str, float | int | str]] = {}
    for subtype in sorted(set(real_labels.tolist())):
        mask = real_labels == subtype
        conf_values = real_confidence[mask]
        unc_values = real_uncertainty[mask]
        subtype_count = int(mask.sum())
        confidence_std = float(np.std(conf_values, ddof=0))
        uncertainty_std = float(np.std(unc_values, ddof=0))
        confidence_scale, confidence_scale_source = calibration_scale(
            subtype_count,
            confidence_std,
            pooled_confidence_std,
            "confidence",
        )
        uncertainty_scale, uncertainty_scale_source = calibration_scale(
            subtype_count,
            uncertainty_std,
            pooled_uncertainty_std,
            "uncertainty",
        )
        references[str(subtype)] = {
            "count": subtype_count,
            "confidence_mean": float(np.mean(conf_values)),
            "confidence_std": confidence_std,
            "confidence_scale": confidence_scale,
            "confidence_scale_source": confidence_scale_source,
            "uncertainty_mean": float(np.mean(unc_values)),
            "uncertainty_std": uncertainty_std,
            "uncertainty_scale": uncertainty_scale,
            "uncertainty_scale_source": uncertainty_scale_source,
        }

    standardized_confidence = np.empty(len(generated_labels), dtype=float)
    standardized_uncertainty = np.empty(len(generated_labels), dtype=float)
    for index, subtype in enumerate(generated_labels):
        reference = references.get(str(int(subtype)))
        if reference is None:
            raise ValueError(
                f"No real subtype reference is available for generated label {subtype}"
            )
        standardized_confidence[index] = (
            generated_confidence[index] - reference["confidence_mean"]
        ) / float(reference["confidence_scale"])
        standardized_uncertainty[index] = (
            generated_uncertainty[index] - reference["uncertainty_mean"]
        ) / float(reference["uncertainty_scale"])

    statistical_violation = (
        standardized_confidence < float(confidence_threshold)
    ) & (standardized_uncertainty < float(uncertainty_threshold))
    return {
        "references": references,
        "standardized_confidence": standardized_confidence,
        "standardized_uncertainty": standardized_uncertainty,
        "statistical_violation": statistical_violation,
    }


class PreparedPidCurator:
    """Fit PID curation once and score independent generated batches."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        schema: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        curation = config["curation"]
        if curation.get("mode") != "subtype_calibrated":
            raise ValueError("PID curation.mode must be subtype_calibrated")
        confidence_threshold = curation.get(
            "standardized_confidence_threshold"
        )
        uncertainty_threshold = curation.get(
            "standardized_uncertainty_threshold"
        )
        if confidence_threshold is None or uncertainty_threshold is None:
            raise ValueError(
                "PID curation requires standardized_confidence_threshold and "
                "standardized_uncertainty_threshold"
            )
        epsilon = float(curation.get("epsilon", PID_CURATION_EPSILON))
        if epsilon != PID_CURATION_EPSILON:
            raise ValueError("PID curation epsilon is fixed at 1e-8 by the method")
        n_estimators = int(curation["n_estimators"])
        if n_estimators < 2:
            raise ValueError("curation.n_estimators must be at least 2")

        self.schema = schema
        self.confidence_threshold = float(confidence_threshold)
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.epsilon = epsilon
        self.n_estimators = n_estimators
        self.encoder = PidFeatureEncoder(schema, curation.get("encoder", {}))
        x_train = self.encoder.fit_transform(train_df)
        label = schema["label_column"]
        self.y_train = train_df[label].to_numpy(dtype=int)
        self.classifier = XGBClassifier(
            n_estimators=n_estimators,
            n_jobs=(
                int(curation["n_jobs"])
                if curation.get("n_jobs") is not None
                else None
            ),
            random_state=int(config["run"]["seed"]),
            objective="multi:softprob",
            num_class=int(schema["n_classes"]),
            eval_metric="mlogloss",
        )
        self.classifier.fit(x_train, self.y_train)

        real_dynamics = self._learning_dynamics(x_train, self.y_train)
        self.real_confidence = real_dynamics.confidence
        self.real_uncertainty = real_dynamics.aleatoric

    def _learning_dynamics(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Curator:
        curator = Curator(x, y)
        for iteration in range(1, self.n_estimators):
            curator.on_iteration_end(self.classifier, iteration)
        return curator

    def score_batch(self, generated_df: pd.DataFrame) -> dict[str, Any]:
        """Score a batch without refitting or using other synthetic rows."""
        label = self.schema["label_column"]
        y_generated = generated_df[label].to_numpy(dtype=int)
        if generated_df.empty:
            generated_confidence = np.asarray([], dtype=float)
            generated_uncertainty = np.asarray([], dtype=float)
        else:
            x_generated = self.encoder.transform(generated_df)
            dynamics = self._learning_dynamics(x_generated, y_generated)
            generated_confidence = dynamics.confidence
            generated_uncertainty = dynamics.aleatoric

        calibrated = subtype_calibrated_scores(
            self.y_train,
            self.real_confidence,
            self.real_uncertainty,
            y_generated,
            generated_confidence,
            generated_uncertainty,
            self.confidence_threshold,
            self.uncertainty_threshold,
            self.epsilon,
        )
        flags = calibrated["statistical_violation"]
        scores: list[dict[str, Any]] = []
        for index in range(len(generated_df)):
            scores.append(
                {
                    "row_index": index,
                    "subtype_id": int(y_generated[index]),
                    "confidence": float(generated_confidence[index]),
                    "aleatoric_uncertainty": float(
                        generated_uncertainty[index]
                    ),
                    "standardized_confidence": float(
                        calibrated["standardized_confidence"][index]
                    ),
                    "standardized_uncertainty": float(
                        calibrated["standardized_uncertainty"][index]
                    ),
                    "statistical_violation": bool(flags[index]),
                }
            )
        return {
            "algorithm": "pid_subtype_calibrated_learning_dynamics",
            "checkpoints": int(self.n_estimators - 1),
            "epsilon": self.epsilon,
            "thresholds": {
                "standardized_confidence": self.confidence_threshold,
                "standardized_uncertainty": self.uncertainty_threshold,
            },
            "encoder": self.encoder.metadata(),
            "references": calibrated["references"],
            "scores": scores,
            "selected": np.where(~flags)[0].astype(int).tolist(),
            "statistical_rejected": np.where(flags)[0].astype(int).tolist(),
        }


def curate_pid_generated(
    train_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    schema: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Fit PID curation once and score a complete generated dataframe."""
    return PreparedPidCurator(train_df, schema, config).score_batch(generated_df)
