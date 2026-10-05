from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import PipelineConfig
from .io_utils import write_json, write_jsonl
from .llm import CompletionClient, LLMResponseError
from .markdown import Block, Chunk, Document, block_map, build_chunks, parse_markdown
from .merge import compact_constraints, deduplicate_constraints
from .model_protocol import BlockReferenceCodec
from .prompts import (
    actor_initial_prompt,
    actor_revision_prompt,
    actor_system_prompt,
    critic_prompt,
    critic_system_prompt,
)
from .schema import (
    CONSTRAINT_TYPES,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    actor_result_schema,
    critic_result_schema,
    json_schema_name,
)
from .statements import add_constraint_statements
from .test_dictionary import configure_scope, filter_constraints_to_scope


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config_hash(config: PipelineConfig) -> str:
    encoded = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class ExtractionOutcome:
    status: str
    constraint_type: str
    chunk_id: str
    actor_rounds: int
    output_candidates: list[dict[str, Any]]
    final_actor_result: dict[str, Any] | None
    final_critic_result: dict[str, Any] | None
    final_feedback: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PreparedDocument:
    document: Document
    chunks: list[Chunk]
    output_directory: Path


@dataclass(slots=True)
class _ExtractionJob:
    index: int
    document: Document
    chunk: Chunk
    constraint_type: str
    blocks: dict[str, Block]
    document_output_directory: Path

    @property
    def result_path(self) -> Path:
        return (
            self.document_output_directory
            / "chunk_results"
            / f"{self.chunk.chunk_id.split(':')[-1]}-{self.constraint_type}.json"
        )


def write_compact_outputs(
    output_directory: Path,
    constraints: list[dict[str, Any]],
) -> dict[str, int]:
    groups, evidence, stats = compact_constraints(constraints)
    write_jsonl(output_directory / "constraints.compact.jsonl", groups)
    write_jsonl(output_directory / "constraint_evidence.jsonl", evidence)
    return stats


def _build_extraction_jobs(prepared: list[PreparedDocument]) -> list[_ExtractionJob]:
    jobs: list[_ExtractionJob] = []
    for item in prepared:
        blocks = block_map(item.document.blocks)
        for chunk in item.chunks:
            for constraint_type in sorted(CONSTRAINT_TYPES):
                jobs.append(
                    _ExtractionJob(
                        index=len(jobs),
                        document=item.document,
                        chunk=chunk,
                        constraint_type=constraint_type,
                        blocks=blocks,
                        document_output_directory=item.output_directory,
                    )
                )
    return jobs


def prepare_documents(
    input_paths: list[Path],
    output_directory: Path,
    config: PipelineConfig,
) -> tuple[list[PreparedDocument], dict[str, Any]]:
    if not input_paths:
        raise ValueError("At least one Markdown input is required")
    output_directory.mkdir(parents=True, exist_ok=True)
    prepared: list[PreparedDocument] = []
    document_ids: set[str] = set()
    for input_path in input_paths:
        if not input_path.is_file():
            raise FileNotFoundError(f"Input Markdown does not exist: {input_path}")
        document = parse_markdown(input_path)
        if document.document_id in document_ids:
            raise ValueError(
                f"Document ID collision for {document.document_id}; place inputs in uniquely named parent directories"
            )
        document_ids.add(document.document_id)
        chunks = build_chunks(
            document,
            max_chars=config.chunk_max_chars,
            overlap_blocks=config.chunk_overlap_blocks,
        )
        document_directory = output_directory / "documents" / document.document_id
        write_json(document_directory / "metadata.json", document.to_metadata())
        write_jsonl(document_directory / "blocks.jsonl", (block.to_dict() for block in document.blocks))
        write_jsonl(document_directory / "chunks.jsonl", (chunk.to_dict() for chunk in chunks))
        prepared.append(PreparedDocument(document, chunks, document_directory))

    primary_ids = [
        block_id
        for item in prepared
        for chunk in item.chunks
        for block_id in chunk.primary_block_ids
    ]
    eligible_ids = [
        block.block_id for item in prepared for block in item.document.blocks if block.eligible
    ]
    if len(primary_ids) != len(set(primary_ids)) or set(primary_ids) != set(eligible_ids):
        raise RuntimeError("Chunk construction failed the exactly-once primary-block coverage invariant")
    manifest = {
        "artifact": "preprocess_manifest",
        "created_at": _utc_now(),
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "config_hash": _config_hash(config),
        "config": config.to_dict(),
        "documents": [
            {
                **item.document.to_metadata(),
                "chunks": len(item.chunks),
                "primary_block_coverage": 1.0,
            }
            for item in prepared
        ],
        "totals": {
            "documents": len(prepared),
            "chunks": sum(len(item.chunks) for item in prepared),
            "eligible_blocks": len(eligible_ids),
            "primary_blocks": len(primary_ids),
        },
    }
    write_json(output_directory / "preprocess_manifest.json", manifest)
    return prepared, manifest


def extract_chunk_type(
    *,
    client: CompletionClient,
    config: PipelineConfig,
    constraint_type: str,
    chunk: Chunk,
    blocks: dict[str, Block],
) -> ExtractionOutcome:
    if constraint_type not in CONSTRAINT_TYPES:
        raise ValueError(f"Unknown constraint type: {constraint_type}")
    references = BlockReferenceCodec.for_chunk(chunk)
    model_chunk = references.model_chunk(chunk)
    actor_schema = actor_result_schema(
        constraint_type,
        model_chunk.primary_block_ids,
        model_chunk.all_block_ids,
    )
    critic_schema = critic_result_schema(
        model_chunk.primary_block_ids,
        model_chunk.all_block_ids,
    )
    trace: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    final_critic: dict[str, Any] | None = None
    feedback: list[dict[str, Any]] = []
    pending_actor_error: str | None = None

    for actor_round in range(1, config.max_rounds + 1):
        user_prompt = (
            actor_initial_prompt(model_chunk, constraint_type)
            if actor_round == 1
            else actor_revision_prompt(
                model_chunk,
                constraint_type,
                references.compact_actor_result(previous),
                references.compact_revision_feedback(feedback),
                actor_round,
            )
        )
        try:
            actor_result = references.expand_actor_result(
                client.complete(
                    model=config.actor_model,
                    system_prompt=actor_system_prompt(constraint_type),
                    user_prompt=user_prompt,
                    schema=actor_schema,
                    schema_name=json_schema_name(f"actor_{constraint_type}"),
                    call_id=f"{chunk.chunk_id}:{constraint_type}:actor:r{actor_round}",
                    request_timeout_seconds=(
                        config.actor_revision_request_timeout_seconds
                        if actor_round > 1
                        else None
                    ),
                    request_retries=(
                        config.actor_revision_request_retries
                        if actor_round > 1
                        else None
                    ),
                    max_output_tokens=(
                        config.actor_revision_max_output_tokens
                        if actor_round > 1
                        else None
                    ),
                )
            )
            pending_actor_error = None
        except LLMResponseError as exc:
            pending_actor_error = f"{type(exc).__name__}: {exc}"
            trace.append(
                {
                    "stage": "actor",
                    "round": actor_round,
                    "status": "response_format_error",
                    "error": pending_actor_error,
                }
            )
            feedback = [
                {
                    "issue_id": "FORMAT1",
                    "category": "schema_error",
                    "candidate_id": None,
                    "block_id": None,
                    "evidence_quote": None,
                    "message": str(exc),
                    "required_action": "Return one schema-conforming JSON object with no surrounding prose.",
                }
            ]
            continue
        except Exception as exc:
            trace.append(
                {
                    "stage": "actor",
                    "round": actor_round,
                    "status": "call_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return ExtractionOutcome(
                status="failed",
                constraint_type=constraint_type,
                chunk_id=chunk.chunk_id,
                actor_rounds=actor_round,
                output_candidates=(previous or {}).get("constraints", []),
                final_actor_result=previous,
                final_critic_result=final_critic,
                final_feedback=feedback,
                trace=trace,
                error=f"Actor call failed: {type(exc).__name__}: {exc}",
            )

        trace.append(
            {
                "stage": "actor",
                "round": actor_round,
                "status": "received",
                "result": actor_result,
            }
        )
        previous = actor_result

        critic_result: dict[str, Any] | None = None
        for critic_attempt in range(config.critic_format_retries + 1):
            model_actor_result = references.compact_actor_result(actor_result)
            if model_actor_result is None:
                raise RuntimeError("Actor result unexpectedly became null")
            critic_user = critic_prompt(model_chunk, constraint_type, model_actor_result)
            try:
                critic_result = references.expand_critic_result(
                    client.complete(
                        model=config.critic_model,
                        system_prompt=critic_system_prompt(constraint_type),
                        user_prompt=critic_user,
                        schema=critic_schema,
                        schema_name=json_schema_name(f"critic_{constraint_type}"),
                        call_id=(
                            f"{chunk.chunk_id}:{constraint_type}:critic:"
                            f"r{actor_round}:format{critic_attempt}"
                        ),
                    )
                )
            except LLMResponseError as exc:
                trace.append(
                    {
                        "stage": "critic",
                        "round": actor_round,
                        "format_attempt": critic_attempt,
                        "status": "response_format_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            except Exception as exc:
                trace.append(
                    {
                        "stage": "critic",
                        "round": actor_round,
                        "format_attempt": critic_attempt,
                        "status": "call_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                return ExtractionOutcome(
                    status="failed",
                    constraint_type=constraint_type,
                    chunk_id=chunk.chunk_id,
                    actor_rounds=actor_round,
                    output_candidates=actor_result.get("constraints", []),
                    final_actor_result=actor_result,
                    final_critic_result=final_critic,
                    final_feedback=feedback,
                    trace=trace,
                    error=f"Critic call failed: {type(exc).__name__}: {exc}",
                )
            trace.append(
                {
                    "stage": "critic",
                    "round": actor_round,
                    "format_attempt": critic_attempt,
                    "status": "received",
                    "result": critic_result,
                }
            )
            break

        if critic_result is None:
            return ExtractionOutcome(
                status="failed",
                constraint_type=constraint_type,
                chunk_id=chunk.chunk_id,
                actor_rounds=actor_round,
                output_candidates=actor_result.get("constraints", []),
                final_actor_result=actor_result,
                final_critic_result=final_critic,
                final_feedback=feedback,
                trace=trace,
                error="Critic response could not be parsed after format retries",
            )
        final_critic = critic_result
        critic_issues = critic_result.get("issues")
        feedback = list(critic_issues) if isinstance(critic_issues, list) else []
        if critic_result.get("verdict") == "pass":
            return ExtractionOutcome(
                status="completed",
                constraint_type=constraint_type,
                chunk_id=chunk.chunk_id,
                actor_rounds=actor_round,
                output_candidates=actor_result.get("constraints", []),
                final_actor_result=actor_result,
                final_critic_result=critic_result,
                final_feedback=[],
                trace=trace,
            )

    if previous is None:
        return ExtractionOutcome(
            status="failed",
            constraint_type=constraint_type,
            chunk_id=chunk.chunk_id,
            actor_rounds=config.max_rounds,
            output_candidates=[],
            final_actor_result=None,
            final_critic_result=final_critic,
            final_feedback=feedback,
            trace=trace,
            error=pending_actor_error or "Actor returned no parseable result",
        )
    return ExtractionOutcome(
        status="failed" if pending_actor_error else "completed",
        constraint_type=constraint_type,
        chunk_id=chunk.chunk_id,
        actor_rounds=config.max_rounds,
        output_candidates=previous.get("constraints", []),
        final_actor_result=previous,
        final_critic_result=final_critic,
        final_feedback=feedback,
        trace=trace,
        error=pending_actor_error,
    )


def run_pipeline(
    *,
    input_paths: list[Path],
    output_directory: Path,
    config: PipelineConfig,
    client: CompletionClient,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    config.validate(require_models=True)
    configure_scope(
        config.ehr_path, config.dictionary_path, config.hpo_path,
        config.hpo_vocabulary_path,
    )
    started_at = _utc_now()
    prepared, preprocess_manifest = prepare_documents(input_paths, output_directory, config)
    direct_constraints: list[dict[str, Any]] = []
    actor_critic_results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    extraction_counts = {"completed": 0, "failed": 0}

    jobs = _build_extraction_jobs(prepared)
    total_jobs = len(jobs)
    completed_jobs = 0
    outcomes: list[ExtractionOutcome | None] = [None] * total_jobs
    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        future_jobs = {
            executor.submit(
                extract_chunk_type,
                client=client,
                config=config,
                constraint_type=job.constraint_type,
                chunk=job.chunk,
                blocks=job.blocks,
            ): job
            for job in jobs
        }
        for future in as_completed(future_jobs):
            job = future_jobs[future]
            outcome = future.result()
            outcomes[job.index] = outcome
            write_json(job.result_path, outcome.to_dict())
            completed_jobs += 1
            if progress:
                progress(
                    f"[{completed_jobs}/{total_jobs}] "
                    f"{job.chunk.chunk_id} {job.constraint_type}"
                )

    for job, outcome in zip(jobs, outcomes, strict=True):
        if outcome is None:
            raise RuntimeError(
                f"Extraction job completed without an outcome: "
                f"{job.chunk.chunk_id} {job.constraint_type}"
            )
        extraction_counts[outcome.status] += 1
        direct_constraints.extend(outcome.output_candidates)
        actor_critic_results.append(
            {
                "document_id": job.document.document_id,
                "chunk_result_path": str(job.result_path.resolve()),
                **outcome.to_dict(),
            }
        )
        if outcome.status == "failed":
            errors.append(
                {
                    "error_type": "actor_critic_execution_failure",
                    "document_id": job.document.document_id,
                    "chunk_id": job.chunk.chunk_id,
                    "constraint_type": job.constraint_type,
                    "message": outcome.error,
                    "chunk_result_path": str(job.result_path.resolve()),
                }
            )

    scoped_constraints, excluded_constraints, scope_filter_stats = (
        filter_constraints_to_scope(direct_constraints)
    )
    final_constraints, deduplication_stats = deduplicate_constraints(scoped_constraints)
    final_constraints = add_constraint_statements(final_constraints)
    write_jsonl(output_directory / "constraints.jsonl", final_constraints)
    write_jsonl(
        output_directory / "scope_excluded_constraints.jsonl",
        excluded_constraints,
    )
    compaction_stats = write_compact_outputs(output_directory, final_constraints)
    write_jsonl(output_directory / "actor_critic_results.jsonl", actor_critic_results)
    write_jsonl(output_directory / "errors.jsonl", errors)
    for obsolete_name in ("critic_passed_constraints.jsonl", "review_queue.jsonl"):
        (output_directory / obsolete_name).unlink(missing_ok=True)

    status = "complete" if not errors else "incomplete"
    manifest = {
        "artifact": "constraint_extraction_run",
        "status": status,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "config_hash": _config_hash(config),
        "config": config.to_dict(),
        "preprocess_manifest": preprocess_manifest,
        "extraction": {
            "jobs": total_jobs,
            **extraction_counts,
        },
        "output": {
            "raw_constraints": len(direct_constraints),
            "in_scope_raw_constraints": len(scoped_constraints),
            "scope_excluded_constraints": len(excluded_constraints),
            "constraints": len(final_constraints),
            "compact_constraint_groups": compaction_stats["constraint_groups"],
            "constraint_evidence_records": compaction_stats[
                "unique_evidence_records"
            ],
            "actor_critic_results": len(actor_critic_results),
            "errors": len(errors),
        },
        "scope_filter": scope_filter_stats,
        "deduplication": deduplication_stats,
        "compaction": compaction_stats,
        "output_mode": "direct_actor_critic",
        "semantic_completion_claim": False,
        "guarantees": {
            "eligible_text_blocks_primary_exactly_once": True,
            "actor_results_locally_validated": False,
            "critic_results_locally_validated": False,
            "evidence_quotes_deterministically_anchored": False,
            "critic_pass_required_for_output": False,
            "final_constraints_semantically_deduplicated": True,
            "test_targets_limited_to_ehr_public": True,
            "test_target_surfaces_verified_against_ehr_public": True,
            "phenotype_targets_limited_to_task_vocabulary": True,
            "phenotype_target_surfaces_verified_against_hpo": True,
            "normal_test_states_and_reference_ranges_excluded": True,
            "constraint_statements_derived_from_structured_fields": True,
            "compact_constraints_normalized": True,
            "potential_conflicts_filtered": False,
            "semantic_precision_or_recall_proven_without_gold": False,
            "input_scope": "markdown_text",
            "actor_and_critic_model_names_differ": config.actor_model != config.critic_model,
        },
    }
    write_json(output_directory / "manifest.json", manifest)
    return manifest
