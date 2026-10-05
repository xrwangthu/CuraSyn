"""OpenAI-compatible serving adapter for Curasyn generation."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx
import numpy as np
import pandas as pd
from openai import OpenAI

from src.data import (
    build_numerical_validity_profile,
    normalize_pid_generated,
    partition_numerically_valid,
)
from src.hpo import load_hpo_ontology, require_hpo_names
from src.generation_checkpoint import (
    append_checkpoint_round,
    load_generation_checkpoint,
    write_checkpoint_state,
)
from src.knowledge import (
    ClinicalKnowledgeBase,
    EmpiricalHpoKnowledge,
    attach_empirical_hpo_residual_probabilities,
    filter_phenotype_knowledge_to_vocabulary,
    merge_empirical_hpo_knowledge,
    render_knowledge,
)
from src.retrieval import (
    build_phenotype_profile,
    build_disease_schedule,
    residual_empirical_hpo_probabilities,
    rows_for_disease,
    select_frequency_aligned_candidates,
    select_random_exemplars,
    select_residual_exemplars,
)
from src.schemas import table_columns
from src.templates import build_pid_generation_prompt


METADATA_VALUES = {
    "integer",
    "float",
    "numeric",
    "categorical",
    "number",
    "No",
    "Yes",
    "continuous",
    "age in years",
    "string",
}


def _checkpoint_identity(
    config: dict[str, Any],
    train_df: pd.DataFrame,
    schema: dict[str, Any],
    schedule: Any,
    knowledge_checksum: str | None,
) -> str:
    operational_generation_keys = {
        "max_rounds",
        "max_stalled_rounds_per_target",
        "request_retries",
        "retry_delay_seconds",
        "api",
    }
    generation = {
        key: value
        for key, value in config["generation"].items()
        if key not in operational_generation_keys
    }
    train_hash = hashlib.sha256(
        pd.util.hash_pandas_object(train_df, index=True).values.tobytes()
    ).hexdigest()
    payload = {
        "protocol": "pid_generation_checkpoint_v1_hpo_filtered",
        "dataset": config.get("data", {}).get("dataset", schema.get("dataset")),
        "seed": int(config["run"]["seed"]),
        "generation": generation,
        "retrieval": config["retrieval"],
        "curation": config.get("curation", {}),
        "knowledge": config.get("knowledge", {}),
        "knowledge_checksum": knowledge_checksum,
        "schema_columns": table_columns(
            schema, include_metadata=True, include_conditions=True
        ),
        "hpo_vocabulary": schema["hpo_vocabulary"],
        "schedule": [
            [disease, int(label), int(count)]
            for (disease, label), count in sorted(schedule.target_counts.items())
        ],
        "train_hash": train_hash,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _dump_target_counter(counter: Counter[tuple[str, int]]) -> list[list[Any]]:
    return [
        [disease, int(label), int(value)]
        for (disease, label), value in sorted(counter.items())
        if value
    ]


def _load_target_counter(values: list[list[Any]]) -> Counter[tuple[str, int]]:
    return Counter(
        {
            (str(disease), int(label)): int(value)
            for disease, label, value in values
        }
    )


@dataclass
class GenerationResult:
    dataframe: pd.DataFrame
    rounds: int
    responses: int
    parsed_rows: int
    valid_rows: int
    parse_failures: int
    request_failures: int
    prompt_preview: str
    final_dataframe: pd.DataFrame | None = None
    curation_passed_rows: int = 0
    curation_rejected_rows: int = 0
    range_rejected_rows: int = 0
    range_violation_counts: dict[str, int] = field(default_factory=dict)
    surplus_rows: int = 0
    retrieval_trace: list[dict[str, Any]] = field(default_factory=list)
    subtype_counts: dict[str, int] = field(default_factory=dict)
    disease_counts: dict[str, int] = field(default_factory=dict)
    knowledge_kb_checksum: str | None = None


def extract_json_objects(text: str) -> tuple[list[dict[str, Any]], int]:
    """Extract flat JSON objects using the same object boundary as CLLM."""
    records: list[dict[str, Any]] = []
    failures = 0
    for candidate in re.findall(r"\{[^{}]*\}", text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            failures += 1
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
        else:
            failures += 1
    return records, failures


def _contains_metadata(record: dict[str, Any]) -> bool:
    return any(isinstance(value, str) and value in METADATA_VALUES for value in record.values())


def _create_client(config: dict[str, Any]) -> Any:
    generation = config["generation"]
    api = generation.get("api", {})
    serving = generation["serving"]

    if serving == "azure_openai":
        from openai import AzureOpenAI

        return AzureOpenAI(
            api_key=api["api_key"],
            azure_endpoint=api["azure_endpoint"],
            api_version=api["api_version"],
        )
    if serving == "vllm":
        for key in ("base_url", "api_key"):
            if not isinstance(api.get(key), str) or not api[key].strip():
                raise ValueError(f"Set generation.api.{key} or its CURASYN_LLM environment override")
        if not isinstance(generation.get("model"), str) or not generation["model"].strip():
            raise ValueError("Set generation.model or CURASYN_LLM_MODEL")
        return OpenAI(
            api_key=api["api_key"],
            base_url=api["base_url"],
            http_client=httpx.Client(trust_env=False),
        )
    if serving in {"together", "openai_compatible"}:
        return OpenAI(api_key=api["api_key"], base_url=api["base_url"])
    raise ValueError(f"Unsupported LLM serving mode: {serving}")


def _messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a tabular synthetic data generation model.",
        },
        {"role": "user", "content": prompt},
    ]


def _response_texts(response: Any) -> list[str]:
    texts = []
    for choice in response.choices:
        message = choice.message
        texts.append(message.content or "")
    return texts


def _report_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    event: str,
    **details: Any,
) -> None:
    if callback is None:
        return
    callback({"event": event, **details, "timestamp": time.time()})


def _combine_generated_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine normalized rows without pandas all-NA dtype inference."""
    if not frames:
        return pd.DataFrame()
    columns = list(frames[0].columns)
    records = [
        record
        for frame in frames
        for record in frame.to_dict(orient="records")
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def generate_pid_with_openai_compatible(
    config: dict[str, Any],
    train_df: pd.DataFrame,
    schema: dict[str, Any],
    client: Any | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    curation_callback: Callable[[pd.DataFrame], dict[str, Any]] | None = None,
    empirical_hpo_knowledge: EmpiricalHpoKnowledge | None = None,
) -> GenerationResult:
    """Generate and immediately curate scheduled disease/Table records."""
    generation = config["generation"]
    retrieval = config["retrieval"]
    if retrieval.get("theta") is None:
        raise ValueError("retrieval.theta is required for residual retrieval")
    knowledge_config = config.get("knowledge", {})
    empirical_config = knowledge_config.get("empirical_hpo", {})
    frequency_control = empirical_config.get("frequency_control", "none")
    if frequency_control not in {"none", "residual"}:
        raise ValueError(
            "knowledge.empirical_hpo.frequency_control must be none or residual"
        )
    if frequency_control == "residual" and empirical_hpo_knowledge is None:
        raise ValueError(
            "Residual empirical HPO control requires mined empirical HPO knowledge"
        )
    knowledge_kb = ClinicalKnowledgeBase(knowledge_config.get("source"))
    hpo_ontology = load_hpo_ontology(knowledge_config["hpo_path"]) if knowledge_config.get("hpo_path") else load_hpo_ontology()
    hpo_names = require_hpo_names(
        schema["hpo_vocabulary"],
        hpo_ontology,
    )
    numerical_validity = build_numerical_validity_profile(
        train_df, schema,
        **config.get("curation", {}).get("numerical_validity", {}),
    )

    target_rows = int(generation["n_samples"])
    seed = int(config["run"]["seed"])
    schedule = build_disease_schedule(
        train_df,
        schema,
        generation.get("disease_weights", {"mode": "empirical"}),
        target_rows,
        seed,
    )
    disease_target_rows: Counter[str] = Counter()
    for (disease_name, _), count in schedule.target_counts.items():
        disease_target_rows[disease_name] += int(count)
    inverse_labels = {
        int(label_id): label_name
        for label_name, label_id in schema["label_mapping"].items()
    }
    max_rounds = int(generation.get("max_rounds", 1000))
    max_stalled_rounds = int(
        generation.get("max_stalled_rounds_per_target", 20)
    )
    if max_stalled_rounds <= 0:
        raise ValueError(
            "generation.max_stalled_rounds_per_target must be positive"
        )
    completions = int(generation.get("n_processes", 2))
    max_retries = int(generation.get("request_retries", 4))
    retry_delay = float(generation.get("retry_delay_seconds", 0))
    rng = np.random.default_rng(seed)
    active_client = client if client is not None else _create_client(config)

    _report_progress(
        progress_callback,
        "pid_generation_schedule",
        total_target_rows=target_rows,
        groups=[
            {
                "disease_name": disease_name,
                "table_id": int(table_id),
                "table_name": inverse_labels[int(table_id)],
                "target_rows": int(count),
            }
            for (disease_name, table_id), count in sorted(
                schedule.target_counts.items()
            )
            if count > 0
        ],
    )

    raw_frames: list[pd.DataFrame] = []
    accepted_frames: list[pd.DataFrame] = []
    accepted_counts: Counter[tuple[str, int]] = Counter()
    curation_passed_rows = 0
    curation_rejected_rows = 0
    range_rejected_rows = 0
    range_violation_counts: Counter[str] = Counter()
    surplus_rows = 0
    raw_valid_rows = 0
    parsed_rows = 0
    parse_failures = 0
    request_failures = 0
    response_count = 0
    prompt_preview = ""
    traces: list[dict[str, Any]] = []
    retrieval_rounds: Counter[tuple[str, int]] = Counter()
    target_attempts: Counter[tuple[str, int]] = Counter()
    target_valid_rows: Counter[tuple[str, int]] = Counter()
    target_passed_rows: Counter[tuple[str, int]] = Counter()
    target_rejected_rows: Counter[tuple[str, int]] = Counter()
    target_range_rejected_rows: Counter[tuple[str, int]] = Counter()
    stalled_rounds: Counter[tuple[str, int]] = Counter()
    rounds_completed = 0
    target_order = list(dict.fromkeys(schedule.targets))
    attempted_in_cycle: set[tuple[str, int]] = set()

    checkpoint_paths = config.get("paths", {})
    checkpoint_state_path = checkpoint_paths.get("checkpoint_state")
    checkpoint_journal_path = checkpoint_paths.get("checkpoint_journal")
    if bool(checkpoint_state_path) != bool(checkpoint_journal_path):
        raise ValueError(
            "paths.checkpoint_state and paths.checkpoint_journal must be set together"
        )
    checkpoint_identity = None
    if checkpoint_state_path and checkpoint_journal_path:
        checkpoint_identity = _checkpoint_identity(
            config,
            train_df,
            schema,
            schedule,
            knowledge_kb.checksum,
        )
        checkpoint = load_generation_checkpoint(
            checkpoint_state_path,
            checkpoint_journal_path,
            expected_identity=checkpoint_identity,
            columns=table_columns(
                schema, include_metadata=True, include_conditions=True
            ),
        )
        if checkpoint is not None:
            state = checkpoint.state
            rounds_completed = int(state["rounds_completed"])
            if rounds_completed > max_rounds:
                raise RuntimeError(
                    "Generation checkpoint has more completed rounds than max_rounds"
                )
            raw_frames = (
                [checkpoint.raw_dataframe]
                if not checkpoint.raw_dataframe.empty
                else []
            )
            accepted_frames = (
                [checkpoint.accepted_dataframe]
                if not checkpoint.accepted_dataframe.empty
                else []
            )
            accepted_counts = _load_target_counter(state["accepted_counts"])
            curation_passed_rows = int(state["curation_passed_rows"])
            curation_rejected_rows = int(state["curation_rejected_rows"])
            range_rejected_rows = int(state["range_rejected_rows"])
            range_violation_counts = Counter(state["range_violation_counts"])
            surplus_rows = int(state["surplus_rows"])
            raw_valid_rows = int(state["raw_valid_rows"])
            parsed_rows = int(state["parsed_rows"])
            parse_failures = int(state["parse_failures"])
            request_failures = int(state["request_failures"])
            response_count = int(state["response_count"])
            prompt_preview = str(state["prompt_preview"])
            traces = checkpoint.traces
            retrieval_rounds = _load_target_counter(state["retrieval_rounds"])
            target_attempts = _load_target_counter(state["target_attempts"])
            target_valid_rows = _load_target_counter(state["target_valid_rows"])
            target_passed_rows = _load_target_counter(state["target_passed_rows"])
            target_rejected_rows = _load_target_counter(
                state["target_rejected_rows"]
            )
            target_range_rejected_rows = _load_target_counter(
                state["target_range_rejected_rows"]
            )
            stalled_rounds = _load_target_counter(state["stalled_rounds"])
            attempted_in_cycle = {
                (str(disease), int(label))
                for disease, label in state["attempted_in_cycle"]
            }
            rng.bit_generator.state = state["rng_state"]
            if set(accepted_counts) - set(target_order):
                raise RuntimeError(
                    "Generation checkpoint contains targets outside the schedule"
                )
            _report_progress(
                progress_callback,
                "pid_generation_resume",
                rounds_completed=rounds_completed,
                raw_rows=raw_valid_rows,
                accepted_rows=int(sum(accepted_counts.values())),
                total_target_rows=target_rows,
                checkpoint_path=str(checkpoint_state_path),
            )

    for round_idx in range(rounds_completed, max_rounds):
        incomplete_targets = [
            target
            for target in target_order
            if accepted_counts[target] < schedule.target_counts[target]
        ]
        if not incomplete_targets:
            break
        eligible_targets = [
            target
            for target in incomplete_targets
            if target not in attempted_in_cycle
        ]
        if not eligible_targets:
            attempted_in_cycle.clear()
            eligible_targets = incomplete_targets
        target = max(
            eligible_targets,
            key=lambda candidate: (
                (
                    schedule.target_counts[candidate]
                    - accepted_counts[candidate]
                )
                / schedule.target_counts[candidate],
                -target_order.index(candidate),
            ),
        )
        attempted_in_cycle.add(target)
        target_disease, target_label = target
        disease_train = rows_for_disease(
            train_df, schema, target_disease, target_label
        )
        deficit = schedule.target_counts[target] - accepted_counts[target]
        disease_size = int(len(disease_train))
        if accepted_frames:
            accepted_output = _combine_generated_frames(accepted_frames)
            accepted_disease = rows_for_disease(
                accepted_output, schema, target_disease
            )
            same_disease_synthetic = rows_for_disease(
                accepted_output, schema, target_disease, target_label
            )
        else:
            accepted_disease = pd.DataFrame(
                columns=table_columns(schema, include_conditions=True)
            )
            same_disease_synthetic = pd.DataFrame(
                columns=table_columns(schema, include_conditions=True)
            )
        phenotype_profile = build_phenotype_profile(
            train_df,
            disease_train,
            same_disease_synthetic,
            schema,
            target_label,
            int(retrieval["theta"]),
            int(knowledge_config.get("max_phenotype_items", 16)),
            max_examination_items=int(
                knowledge_config.get("max_examination_items", 4)
            ),
            numerical_validity=numerical_validity,
            minimum_support=int(retrieval.get("coverage_minimum_support", 2)),
            max_shared_anchors=int(retrieval.get("max_shared_anchors", 4)),
        )
        use_residual = (
            disease_size >= int(retrieval["theta"])
            and not same_disease_synthetic.empty
        )
        if use_residual:
            selection = select_residual_exemplars(
                disease_train,
                same_disease_synthetic,
                schema,
                target_label,
                int(retrieval["max_exemplars"]),
                int(retrieval.get("numerical_bins", 50)),
                int(retrieval_rounds[target]),
                rng,
                retrieval.get("numerical_missing_policy"),
            )
            retrieval_rounds[target] += 1
            selection.trace["mode"] = "same_disease_residual"
        else:
            selection = select_random_exemplars(
                disease_train,
                schema,
                target_label,
                int(retrieval["max_exemplars"]),
                rng,
            )
            selection.trace["mode"] = "same_disease_random"
            if disease_size < int(retrieval["theta"]):
                fallback_reason = "disease_below_theta"
            else:
                fallback_reason = "no_prior_same_disease_synthetic_rows"
            selection.trace["fallback_reason"] = fallback_reason
        disease_column = schema["disease_name_column"]
        selection.trace["target_disease_name"] = target_disease
        selection.trace["phenotype_profile"] = phenotype_profile.trace()
        knowledge = knowledge_kb.retrieve(
            [target_disease],
            max_items=int(knowledge_config.get("max_items", 12)),
            max_phenotype_items=int(
                knowledge_config.get("max_phenotype_items", 16)
            ),
            max_examination_items=int(
                knowledge_config.get("max_examination_items", 4)
            ),
            include_demographic=bool(
                knowledge_config.get("include_demographic", True)
            ),
        )
        knowledge = merge_empirical_hpo_knowledge(
            knowledge,
            empirical_hpo_knowledge,
            hpo_names,
            max_items=int(knowledge_config.get("max_phenotype_items", 16)),
        )
        knowledge, filtered_knowledge_ids = (
            filter_phenotype_knowledge_to_vocabulary(
                knowledge,
                schema["hpo_vocabulary"],
                hpo_ontology.canonical_ids,
            )
        )
        residual_probabilities: dict[str, float] = {}
        if frequency_control == "residual":
            empirical_frequencies = {
                item["target_code"]: float(item["empirical_frequency"])
                for item in knowledge.knowledge
                if item.get("empirical_frequency") is not None
            }
            residual_probabilities = residual_empirical_hpo_probabilities(
                empirical_frequencies,
                accepted_disease,
                schema,
                int(disease_target_rows[target_disease]),
            )
            knowledge = attach_empirical_hpo_residual_probabilities(
                knowledge, residual_probabilities
            )
        knowledge_trace = knowledge.trace()
        knowledge_trace["filtered_out_of_vocabulary_phenotype_ids"] = (
            filtered_knowledge_ids
        )
        knowledge_trace["filtered_out_of_vocabulary_phenotype_count"] = len(
            filtered_knowledge_ids
        )
        prompt = build_pid_generation_prompt(
            selection.dataframe,
            target_disease,
            schema,
            hpo_names=hpo_names,
            coverage_profile=phenotype_profile,
            background_knowledge=render_knowledge(knowledge),
            background_hpo_codes={
                item["target_code"]
                for item in knowledge.knowledge
                if item.get("empirical_frequency") is not None
            },
        )
        if not prompt_preview:
            prompt_preview = prompt[:1000]

        attempt = 0
        round_completions = min(completions, deficit)
        while True:
            _report_progress(
                progress_callback,
                "pid_generation_request",
                round=round_idx + 1,
                max_rounds=max_rounds,
                attempt=attempt + 1,
                max_retries=max_retries,
                completions=round_completions,
                target_disease_name=target_disease,
                scheduled_table_id=int(target_label),
                target_subtype_name=inverse_labels[target_label],
                accepted_for_target=int(accepted_counts[target]),
                target_rows=int(schedule.target_counts[target]),
                raw_rows=int(raw_valid_rows),
                accepted_rows_total=int(sum(accepted_counts.values())),
                total_target_rows=target_rows,
                retrieval_mode=selection.trace["mode"],
                fallback_reason=selection.trace.get("fallback_reason"),
            )
            try:
                response = active_client.chat.completions.create(
                    model=generation["model"],
                    messages=_messages(prompt),
                    temperature=float(generation["temperature"]),
                    max_tokens=int(generation["max_tokens"]),
                    top_p=float(generation.get("top_p", 0.95)),
                    n=round_completions,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                )
                break
            except Exception as error:
                request_failures += 1
                attempt += 1
                will_retry = attempt < max_retries
                next_completions = max(1, round_completions // 2)
                _report_progress(
                    progress_callback,
                    "pid_generation_retry",
                    round=round_idx + 1,
                    target_disease_name=target_disease,
                    scheduled_table_id=int(target_label),
                    target_subtype_name=inverse_labels[target_label],
                    attempt=attempt,
                    max_retries=max_retries,
                    will_retry=will_retry,
                    next_completions=next_completions,
                    request_failures=request_failures,
                    error=f"{type(error).__name__}: {error}",
                )
                if attempt >= max_retries:
                    raise
                round_completions = next_completions
                if retry_delay > 0:
                    time.sleep(retry_delay)

        round_records: list[dict[str, Any]] = []
        texts = _response_texts(response)
        response_count += len(texts)
        rounds_completed = round_idx + 1
        for text in texts:
            parsed, failures = extract_json_objects(text)
            parse_failures += failures
            parsed_rows += len(parsed)
            round_records.extend(
                record for record in parsed if not _contains_metadata(record)
            )
        normalized = normalize_pid_generated(round_records, schema, target_label)
        if not normalized.empty:
            normalized.insert(
                0,
                schema["record_id_column"],
                [
                    f"syn_{seed}_{index:06d}"
                    for index in range(
                        raw_valid_rows + 1,
                        raw_valid_rows + len(normalized) + 1,
                    )
                ],
            )
            normalized[disease_column] = json.dumps(
                [target_disease], ensure_ascii=False, separators=(",", ":")
            )
            normalized = normalized[
                table_columns(
                    schema,
                    include_metadata=True,
                    include_conditions=True,
                )
            ]
            raw_frames.append(normalized)
            raw_valid_rows += len(normalized)

        numerical_partition = partition_numerically_valid(
            normalized, numerical_validity
        )
        gate_positions = numerical_partition.valid_indices
        range_rejected_indices = numerical_partition.rejected_indices
        gate_valid = normalized.iloc[gate_positions].reset_index(drop=True)
        range_rejected_rows += len(range_rejected_indices)
        range_violation_counts.update(numerical_partition.violation_counts)

        if gate_valid.empty:
            selected_indices: list[int] = []
            rejected_indices: list[int] = []
        elif curation_callback is None:
            selected_indices = list(gate_positions)
            rejected_indices = []
        else:
            curation = curation_callback(gate_valid)
            local_selected = [int(index) for index in curation["selected"]]
            local_rejected = [
                int(index) for index in curation["statistical_rejected"]
            ]
            if sorted(local_selected + local_rejected) != list(
                range(len(gate_valid))
            ):
                raise RuntimeError(
                    "Online curation partitions do not cover the generated batch"
                )
            selected_indices = [gate_positions[index] for index in local_selected]
            rejected_indices = [gate_positions[index] for index in local_rejected]

        passed = normalized.iloc[selected_indices].reset_index(drop=True)
        frequency_selection = {
            "applied": False,
            "candidate_rows": int(len(passed)),
            "selected_passed_indices": list(range(min(deficit, len(passed)))),
            "target_probabilities": {},
        }
        if frequency_control == "residual":
            accepted, frequency_selection = select_frequency_aligned_candidates(
                passed,
                schema,
                residual_probabilities,
                deficit,
            )
        else:
            accepted = passed.head(deficit).reset_index(drop=True)
        round_surplus = len(passed) - len(accepted)
        curation_passed_rows += len(passed)
        curation_rejected_rows += len(rejected_indices)
        surplus_rows += round_surplus
        target_attempts[target] += 1
        target_valid_rows[target] += len(normalized)
        target_passed_rows[target] += len(passed)
        target_rejected_rows[target] += len(rejected_indices)
        target_range_rejected_rows[target] += len(range_rejected_indices)
        if not accepted.empty:
            accepted_frames.append(accepted)
            accepted_counts[target] += len(accepted)
            stalled_rounds[target] = 0
        else:
            stalled_rounds[target] += 1

        trace = {
            "round": rounds_completed,
            **selection.trace,
            "scheduled_table_id": int(target_label),
            "target_subtype_name": inverse_labels[target_label],
            "batch_valid_rows": int(len(normalized)),
            "curation_passed_rows": int(len(passed)),
            "curation_rejected_rows": int(len(rejected_indices)),
            "range_rejected_rows": int(len(range_rejected_indices)),
            "range_violation_counts": numerical_partition.violation_counts,
            "surplus_rows": int(round_surplus),
            "accepted_rows": int(len(accepted)),
            "target_rows": int(schedule.target_counts[target]),
            "accepted_for_target": int(accepted_counts[target]),
            "target_attempts": int(target_attempts[target]),
            "target_valid_rows": int(target_valid_rows[target]),
            "target_curation_passed_rows": int(target_passed_rows[target]),
            "target_curation_rejected_rows": int(
                target_rejected_rows[target]
            ),
            "target_range_rejected_rows": int(
                target_range_rejected_rows[target]
            ),
            "stalled_rounds_for_target": int(stalled_rounds[target]),
            "knowledge": knowledge_trace,
        }
        if frequency_control == "residual":
            trace["empirical_hpo_frequency_control"] = {
                "mode": "residual",
                "disease_target_rows": int(disease_target_rows[target_disease]),
                "disease_accepted_rows_before": int(len(accepted_disease)),
                "disease_remaining_rows_before": int(
                    disease_target_rows[target_disease] - len(accepted_disease)
                ),
                **frequency_selection,
            }
        traces.append(trace)
        _report_progress(
            progress_callback,
            "pid_generation_round",
            **trace,
            max_rounds=max_rounds,
            total_target_rows=target_rows,
            responses=response_count,
            raw_rows=int(raw_valid_rows),
            accepted_rows_total=int(sum(accepted_counts.values())),
            parsed_rows=parsed_rows,
            parse_failures=parse_failures,
            request_failures=request_failures,
        )
        if checkpoint_identity is not None:
            append_checkpoint_round(
                checkpoint_journal_path,
                {
                    "round": rounds_completed,
                    "raw_records": json.loads(
                        normalized.to_json(orient="records")
                    ),
                    "accepted_records": json.loads(
                        accepted.to_json(orient="records")
                    ),
                    "trace": trace,
                },
            )
            write_checkpoint_state(
                checkpoint_state_path,
                {
                    "identity": checkpoint_identity,
                    "rounds_completed": rounds_completed,
                    "raw_valid_rows": raw_valid_rows,
                    "accepted_rows": int(sum(accepted_counts.values())),
                    "accepted_counts": _dump_target_counter(accepted_counts),
                    "curation_passed_rows": curation_passed_rows,
                    "curation_rejected_rows": curation_rejected_rows,
                    "range_rejected_rows": range_rejected_rows,
                    "range_violation_counts": dict(range_violation_counts),
                    "surplus_rows": surplus_rows,
                    "parsed_rows": parsed_rows,
                    "parse_failures": parse_failures,
                    "request_failures": request_failures,
                    "response_count": response_count,
                    "prompt_preview": prompt_preview,
                    "retrieval_rounds": _dump_target_counter(retrieval_rounds),
                    "target_attempts": _dump_target_counter(target_attempts),
                    "target_valid_rows": _dump_target_counter(target_valid_rows),
                    "target_passed_rows": _dump_target_counter(target_passed_rows),
                    "target_rejected_rows": _dump_target_counter(
                        target_rejected_rows
                    ),
                    "target_range_rejected_rows": _dump_target_counter(
                        target_range_rejected_rows
                    ),
                    "stalled_rounds": _dump_target_counter(stalled_rounds),
                    "attempted_in_cycle": [
                        [disease, int(label)]
                        for disease, label in sorted(attempted_in_cycle)
                    ],
                    "rng_state": rng.bit_generator.state,
                },
            )
        if stalled_rounds[target] >= max_stalled_rounds:
            raise RuntimeError(
                "PID generation stalled for "
                f"disease={target_disease}, "
                f"table={inverse_labels[target_label]}: "
                f"{stalled_rounds[target]} consecutive target attempts "
                "without an admitted row; "
                f"quota={accepted_counts[target]}/"
                f"{schedule.target_counts[target]}, "
                f"total_target_attempts={target_attempts[target]}, "
                f"valid={target_valid_rows[target]}, "
                f"range_rejected={target_range_rejected_rows[target]}, "
                f"passed={target_passed_rows[target]}, "
                f"rejected={target_rejected_rows[target]}"
            )

    accepted_rows = int(sum(accepted_counts.values()))
    if accepted_rows < target_rows:
        deficits = {
            f"{disease}|{inverse_labels[label]}": int(
                schedule.target_counts[(disease, label)]
                - accepted_counts[(disease, label)]
            )
            for disease, label in target_order
            if accepted_counts[(disease, label)]
            < schedule.target_counts[(disease, label)]
        }
        raise RuntimeError(
            "PID generation accepted "
            f"{accepted_rows} rows after {rounds_completed} rounds; "
            f"requested {target_rows}; remaining deficits={deficits}"
        )
    raw_output = _combine_generated_frames(raw_frames)[
        table_columns(schema, include_metadata=True, include_conditions=True)
    ]
    final_output = _combine_generated_frames(accepted_frames)[
        table_columns(schema, include_metadata=True, include_conditions=True)
    ]
    return GenerationResult(
        dataframe=raw_output,
        rounds=rounds_completed,
        responses=response_count,
        parsed_rows=parsed_rows,
        valid_rows=len(raw_output),
        parse_failures=parse_failures,
        request_failures=request_failures,
        prompt_preview=prompt_preview,
        final_dataframe=final_output,
        curation_passed_rows=int(curation_passed_rows),
        curation_rejected_rows=int(curation_rejected_rows),
        range_rejected_rows=int(range_rejected_rows),
        range_violation_counts=dict(range_violation_counts),
        surplus_rows=int(surplus_rows),
        retrieval_trace=traces,
        subtype_counts={
            str(label): int(
                sum(
                    count
                    for (_, mapped), count in accepted_counts.items()
                    if mapped == label
                )
            )
            for label in sorted({label for _, label in schedule.weights})
        },
        disease_counts={
            name: int(
                sum(
                    count
                    for (target_name, _), count in accepted_counts.items()
                    if target_name == name
                )
            )
            for name in sorted(schedule.disease_to_labels)
        },
        knowledge_kb_checksum=knowledge_kb.checksum,
    )
