"""End-to-end Curasyn generation and statistical curation runner."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.curation import PreparedPidCurator
from src.data import (
    load_split,
    resolve_dataset_config,
)
from src.generation_checkpoint import clear_generation_checkpoint
from src.llm import generate_pid_with_openai_compatible
from src.knowledge import (
    ClinicalKnowledgeBase,
    EmpiricalHpoKnowledge,
    mine_empirical_hpo_relations,
)
from src.runtime_checks import check_writable_outputs
from src.retrieval import (
    disease_to_label_mapping,
    resolve_disease_weights,
)


def _progress(stage: str, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{stage}] {message}", flush=True)


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{name}={count}" for name, count in counts.items())


def _pipeline_progress(
    phase: str,
    raw_rows: int,
    final_rows: int,
    final_target_rows: int,
    *,
    status: str,
    details: str | None = None,
) -> None:
    parts = [
        f"phase={phase}",
        f"raw={raw_rows}",
        f"final={final_rows}/{final_target_rows}",
        f"status={status}",
    ]
    if details:
        parts.append(details)
    _progress("PROGRESS", " ".join(parts))


def _generation_progress(
    record: dict[str, Any],
) -> None:
    event = record["event"]
    if event == "pid_generation_schedule":
        groups = record["groups"]
        _progress(
            "SCHEDULE",
            f"groups={len(groups)} total_target_rows={record['total_target_rows']}",
        )
        for group in groups:
            _progress(
                "SCHEDULE",
                "disease={disease_name} table={table_name} "
                "target_rows={target_rows}".format(
                    **group
                ),
            )
        return

    if event == "pid_generation_resume":
        _progress(
            "RESUME",
            "round={rounds_completed} raw={raw_rows} final={accepted_rows}/"
            "{total_target_rows} checkpoint={checkpoint_path}".format(**record),
        )
        return

    if event == "pid_generation_request":
        fallback = record.get("fallback_reason")
        fallback_text = f" fallback={fallback}" if fallback else ""
        _progress(
            "GENERATION",
            "round={round}/{max_rounds} request attempt={attempt}/{max_retries} "
            "disease={target_disease_name} table={target_subtype_name} "
            "completions={completions} retrieval={retrieval_mode} "
            "final={accepted_rows_total}/{total_target_rows} "
            "quota={accepted_for_target}/{target_rows}"
            f"{fallback_text}".format(**record),
        )
        return

    if event == "pid_generation_retry":
        target = (
            f" disease={record['target_disease_name']}"
            f" table={record['target_subtype_name']}"
        )
        action = "retrying" if record["will_retry"] else "giving_up"
        _progress(
            "RETRY",
            f"round={record['round']}{target} attempt={record['attempt']}/"
            f"{record['max_retries']} action={action} "
            f"next_completions={record['next_completions']} "
            f"error={record['error']}",
        )
        return

    if event == "pid_generation_round":
        details = (
            "round={round}/{max_rounds} disease={target_disease_name} "
            "table={target_subtype_name} batch_valid={batch_valid_rows} "
            f"range_rejected={int(record.get('range_rejected_rows', 0))} "
            "passed={curation_passed_rows} "
            "rejected={curation_rejected_rows} admitted={accepted_rows} "
            "quota={accepted_for_target}/{target_rows}".format(**record)
        )
        if record["surplus_rows"]:
            details += f" surplus={record['surplus_rows']}"
        if record["parse_failures"]:
            details += f" parse_failures={record['parse_failures']}"
        if record["request_failures"]:
            details += f" request_failures={record['request_failures']}"
        fallback = record.get("fallback_reason")
        if fallback:
            details += f" fallback={fallback}"
        _pipeline_progress(
            "generation",
            int(record["raw_rows"]),
            int(record["accepted_rows_total"]),
            int(record["total_target_rows"]),
            status="running",
            details=details,
        )


def _empirical_hpo_minimum_support(config: dict[str, Any]) -> int:
    empirical = config.get("knowledge", {}).get("empirical_hpo", {})
    value = empirical.get("minimum_support")
    dataset = str(config.get("data", {}).get("dataset", ""))
    if isinstance(value, dict):
        if dataset not in value:
            raise ValueError(
                "Enabled knowledge.empirical_hpo.minimum_support has no value "
                f"for dataset {dataset!r}"
            )
        value = value[dataset]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            "Enabled knowledge.empirical_hpo requires minimum_support to be "
            "a positive integer or a dataset-to-positive-integer mapping"
        )
    return value


def _validate_curasyn_config(config: dict[str, Any]) -> None:
    if config["schema"].get("dataset") != "ehr":
        raise ValueError("Curasyn requires an EHR semantic schema")
    generation = config["generation"]
    retrieval = config["retrieval"]
    if int(generation["n_samples"]) <= 0:
        raise ValueError("generation.n_samples must be positive")
    if retrieval.get("max_exemplars") is None:
        raise ValueError("retrieval.max_exemplars is required for Curasyn generation")
    if int(retrieval["max_exemplars"]) <= 0:
        raise ValueError("retrieval.max_exemplars must be positive")
    if retrieval.get("theta") is None:
        raise ValueError("retrieval.theta is required for residual retrieval")
    if int(retrieval["theta"]) <= 0:
        raise ValueError("retrieval.theta must be positive")
    if retrieval.get("numerical_missing_policy") not in {"error", "median"}:
        raise ValueError(
            "retrieval.numerical_missing_policy must be error or median"
        )
    resolve_disease_weights(
        load_split(
            config["paths"]["train_csv"],
            config["schema"],
            include_conditions=True,
        ),
        config["schema"],
        generation.get("disease_weights", {}),
    )
    knowledge = config.get("knowledge", {})
    knowledge_defaults = {
        "max_items": 12,
        "max_phenotype_items": 16,
        "max_examination_items": 4,
    }
    for key, default in knowledge_defaults.items():
        if int(knowledge.get(key, default)) < (1 if key == "max_items" else 0):
            raise ValueError(f"knowledge.{key} has an invalid retrieval limit")
    empirical = knowledge.get("empirical_hpo", {})
    enabled = empirical.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("knowledge.empirical_hpo.enabled must be boolean")
    frequency_control = empirical.get("frequency_control", "none")
    if frequency_control not in {"none", "residual"}:
        raise ValueError(
            "knowledge.empirical_hpo.frequency_control must be none or residual"
        )
    if frequency_control != "none" and not enabled:
        raise ValueError(
            "knowledge.empirical_hpo.frequency_control requires empirical_hpo.enabled"
        )
    if enabled:
        _empirical_hpo_minimum_support(config)
    ClinicalKnowledgeBase(knowledge.get("source"))
    curation = config["curation"]
    if curation.get("standardized_confidence_threshold") is None:
        raise ValueError("curation.standardized_confidence_threshold is required")
    if curation.get("standardized_uncertainty_threshold") is None:
        raise ValueError("curation.standardized_uncertainty_threshold is required")
    # Construction validates every encoder choice before an expensive run.
    from src.curation import PidFeatureEncoder

    PidFeatureEncoder(config["schema"], curation.get("encoder", {}))


def _write_frame(path: str | Path, dataframe: pd.DataFrame) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_curasyn(
    config: dict[str, Any],
    client: Any | None = None,
) -> dict[str, Any]:
    """Run closed-loop Curasyn generation with online statistical curation."""
    config = resolve_dataset_config(config)
    schema = config["schema"]
    paths = config["paths"]
    empirical_config = config.get("knowledge", {}).get("empirical_hpo", {})
    empirical_enabled = empirical_config.get("enabled", False) is True
    if empirical_enabled:
        artifact_name = config["artifact_name"]
        result_directory = Path(paths["effective_csv"]).parent
        paths["empirical_hpo_csv"] = str(
            result_directory / f"{artifact_name}_empirical_hpo.csv"
        )
        paths["empirical_hpo_manifest"] = str(
            result_directory / f"{artifact_name}_empirical_hpo_manifest.json"
        )
    output_files = [paths["result_csv"], paths["effective_csv"]]
    if empirical_enabled:
        output_files.extend(
            [paths["empirical_hpo_csv"], paths["empirical_hpo_manifest"]]
        )
    checkpoint_state = paths.get("checkpoint_state")
    checkpoint_journal = paths.get("checkpoint_journal")
    if bool(checkpoint_state) != bool(checkpoint_journal):
        raise ValueError(
            "paths.checkpoint_state and paths.checkpoint_journal must be set together"
        )
    if checkpoint_state and checkpoint_journal:
        output_files.extend([checkpoint_state, checkpoint_journal])
    check_writable_outputs(files=output_files)
    started = time.time()
    dataset = config["data"]["dataset"]
    generation_config = config["generation"]
    retrieval_config = config["retrieval"]
    curation_config = config["curation"]
    final_target = int(generation_config["n_samples"])
    _progress(
        "START",
        f"dataset={dataset} method=curasyn "
        f"artifact={config['artifact_name']}",
    )
    vocabulary_source = schema["hpo_vocabulary_source"]
    _progress(
        "CONFIG",
        f"hpo_vocabulary={vocabulary_source['mode']} "
        f"path={vocabulary_source['path']} "
        f"origin={vocabulary_source['origin']} "
        f"codes={len(schema['hpo_vocabulary'])}",
    )
    _progress(
        "CONFIG",
        f"serving={generation_config['serving']} "
        f"model={generation_config['model']} "
        f"base_url={generation_config.get('api', {}).get('base_url')} "
        f"final_target={final_target} "
        f"max_rounds={int(generation_config.get('max_rounds', 1000))} "
        f"completions={int(generation_config['n_processes'])} "
        f"temperature={float(generation_config['temperature'])} "
        f"max_tokens={int(generation_config['max_tokens'])}",
    )
    _progress(
        "CONFIG",
        f"retrieval=residual "
        f"max_exemplars={int(retrieval_config['max_exemplars'])} "
        f"theta={retrieval_config.get('theta')} "
        f"missing_policy={retrieval_config.get('numerical_missing_policy')}",
    )
    _progress(
        "CONFIG",
        f"curation=online "
        f"numerical_validity_gate=train_only_iqr "
        f"confidence_threshold="
        f"{curation_config.get('standardized_confidence_threshold')} "
        f"uncertainty_threshold="
        f"{curation_config.get('standardized_uncertainty_threshold')}",
    )
    _progress("PREPARE", "validating configuration, data, and knowledge base")
    _validate_curasyn_config(config)
    _progress("PREPARE", "validation complete")

    _progress("PREPARE", f"loading training split: {paths['train_csv']}")
    train_df = load_split(
        paths["train_csv"],
        schema,
        include_metadata=True,
        include_conditions=True,
    )
    empirical_knowledge = None
    empirical_record: dict[str, Any] = {"enabled": False}
    if empirical_enabled:
        minimum_support = _empirical_hpo_minimum_support(config)
        relations = mine_empirical_hpo_relations(
            train_df,
            schema,
            minimum_support,
        )
        empirical_knowledge = EmpiricalHpoKnowledge(relations)
        _write_frame(paths["empirical_hpo_csv"], relations)
        manifest = {
            "schema_version": 1,
            "source": "train",
            "train_csv": str(paths["train_csv"]),
            "train_sha256": _sha256(paths["train_csv"]),
            "minimum_support": minimum_support,
            "frequency_definition": "records_with_disease_and_hpo / records_with_disease",
            "disease_count": int(relations["disease"].nunique()),
            "relation_count": int(len(relations)),
            "relations_csv": str(paths["empirical_hpo_csv"]),
            "relations_sha256": _sha256(paths["empirical_hpo_csv"]),
        }
        _write_json(paths["empirical_hpo_manifest"], manifest)
        empirical_record = {
            "enabled": True,
            "minimum_support": minimum_support,
            "disease_count": manifest["disease_count"],
            "relation_count": manifest["relation_count"],
            "relations_csv": str(paths["empirical_hpo_csv"]),
            "manifest": str(paths["empirical_hpo_manifest"]),
        }
        _progress(
            "KNOWLEDGE",
            f"empirical_hpo diseases={manifest['disease_count']} "
            f"relations={manifest['relation_count']} "
            f"minimum_support={minimum_support}",
        )
    paths.setdefault(
        "dataset_csv",
        str(
            config.get("data", {}).get(
                "processed_csv",
                f"datasets/{config['data']['dataset']}/processed/ehr.csv",
            )
        ),
    )
    disease_weights, train_disease_mapping = resolve_disease_weights(
        train_df,
        schema,
        generation_config.get("disease_weights", {}),
    )
    canonical_df = load_split(
        paths["dataset_csv"], schema, include_conditions=True
    )
    disease_mapping = disease_to_label_mapping(canonical_df, schema)
    if any(
        not set(labels).issubset(set(disease_mapping.get(name, [])))
        for name, labels in train_disease_mapping.items()
    ):
        raise ValueError("D_train disease mapping differs from canonical EHR mapping")
    _progress(
        "DATA",
        f"training_rows={len(train_df)} canonical_rows={len(canonical_df)} "
        f"diseases={len(train_disease_mapping)} "
        f"disease_table_targets={len(disease_weights)} mapping=validated",
    )
    _progress(
        "CURATION",
        "fitting train-only classifier and subtype references",
    )
    prepared_curator = PreparedPidCurator(train_df, schema, config)
    _progress(
        "CURATION",
        f"ready checkpoints={prepared_curator.n_estimators - 1} "
        f"training_rows={len(train_df)}",
    )

    run_record: dict[str, Any] = {
        "sysname": config["sysname"],
        "method": "Curasyn",
        "dataset": dataset,
        "split_name": config["data"]["split_name"],
        "seed": int(config["run"]["seed"]),
        "artifact_tag": config.get("run", {}).get("artifact_tag"),
        "result_csv": paths["result_csv"],
        "empirical_hpo": empirical_record,
    }

    _pipeline_progress(
        "generation",
        0,
        0,
        final_target,
        status="running",
    )
    _progress("GENERATION", "LLM generation started")
    generation = generate_pid_with_openai_compatible(
        config,
        train_df,
        schema,
        client=client,
        progress_callback=_generation_progress,
        curation_callback=prepared_curator.score_batch,
        empirical_hpo_knowledge=empirical_knowledge,
    )
    raw_df = generation.dataframe
    final_df = generation.final_dataframe
    if final_df is None:
        raise RuntimeError("PID generation did not return an online-curated dataframe")
    if len(final_df) != final_target:
        raise RuntimeError(
            f"Online curation returned {len(final_df)} rows; "
            f"expected {final_target}"
        )
    record_id_column = schema["record_id_column"]
    if raw_df[record_id_column].duplicated().any():
        raise RuntimeError("Raw LLM record IDs must be unique")
    if not set(final_df[record_id_column]).issubset(set(raw_df[record_id_column])):
        raise RuntimeError("Final record IDs must be a subset of raw LLM record IDs")
    if (
        generation.range_rejected_rows
        + generation.curation_passed_rows
        + generation.curation_rejected_rows
        != len(raw_df)
    ):
        raise RuntimeError(
            "Numerical gate and online curation counts do not cover raw LLM rows"
        )
    if generation.curation_passed_rows != len(final_df) + generation.surplus_rows:
        raise RuntimeError("Online curation passed rows do not match final plus surplus")

    _write_frame(paths["result_csv"], raw_df)
    _write_frame(paths["effective_csv"], final_df)
    if checkpoint_state and checkpoint_journal:
        clear_generation_checkpoint(checkpoint_state, checkpoint_journal)
    fallback_counts = Counter(
        trace.get("fallback_reason")
        for trace in generation.retrieval_trace
        if trace.get("fallback_reason") is not None
    )
    run_record["generation"] = {
        "rounds": generation.rounds,
        "responses": generation.responses,
        "parsed_rows": generation.parsed_rows,
        "valid_rows": generation.valid_rows,
        "accepted_rows": len(final_df),
        "parse_failures": generation.parse_failures,
        "request_failures": generation.request_failures,
        "subtype_counts": generation.subtype_counts,
        "disease_counts": generation.disease_counts,
        "knowledge_kb_checksum": generation.knowledge_kb_checksum,
        "fallback_counts": dict(fallback_counts),
    }
    _progress(
        "GENERATION",
        f"complete raw={generation.valid_rows} final={len(final_df)} "
        f"rounds={generation.rounds} "
        f"responses={generation.responses} parsed={generation.parsed_rows} "
        f"parse_failures={generation.parse_failures} "
        f"request_failures={generation.request_failures} "
        f"range_rejected={generation.range_rejected_rows}",
    )
    nonzero_disease_counts = {
        name: count
        for name, count in generation.disease_counts.items()
        if count > 0
    }
    if nonzero_disease_counts:
        _progress(
            "GENERATION",
            f"disease_counts: {_format_counts(nonzero_disease_counts)}",
        )
    if fallback_counts:
        _progress(
            "RETRIEVAL",
            f"fallback_counts: {_format_counts(dict(fallback_counts))}",
        )
    curation_counts = {
        "raw_valid": len(raw_df),
        "range_rejected": generation.range_rejected_rows,
        "passed": generation.curation_passed_rows,
        "rejected": generation.curation_rejected_rows,
        "surplus": generation.surplus_rows,
        "accepted": len(final_df),
    }
    _progress("CURATION", f"online complete {_format_counts(curation_counts)}")
    if generation.range_violation_counts:
        _progress(
            "VALIDITY",
            "range_rejected_by_field: "
            f"{_format_counts(generation.range_violation_counts)}",
        )
    run_record["curation"] = {
        "counts": curation_counts,
        "final_csv": paths["effective_csv"],
        "final_rows": len(final_df),
    }
    _progress(
        "OUTPUT",
        f"llm_rows={len(raw_df)} path={paths['result_csv']}",
    )
    _progress(
        "OUTPUT",
        f"final_rows={len(final_df)} path={paths['effective_csv']}",
    )
    _pipeline_progress(
        "complete",
        len(raw_df),
        len(final_df),
        final_target,
        status="complete",
    )

    run_record["seconds"] = time.time() - started
    _progress("DONE", f"elapsed_seconds={run_record['seconds']:.1f}")
    return run_record
