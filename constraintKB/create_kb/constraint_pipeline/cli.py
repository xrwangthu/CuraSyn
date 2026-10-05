from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .evaluation import evaluate_constraints
from .io_utils import read_json, read_jsonl, write_json
from .llm import OpenAICompatibleClient
from .pipeline import prepare_documents, run_pipeline, write_compact_outputs


def _load_config(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig.from_file(args.config) if args.config else PipelineConfig()
    for argument, field_name in (
        ("actor_model", "actor_model"),
        ("critic_model", "critic_model"),
        ("base_url", "base_url"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            setattr(config, field_name, value)
    if getattr(args, "chunk_max_chars", None) is not None:
        config.chunk_max_chars = args.chunk_max_chars
    if getattr(args, "chunk_overlap_blocks", None) is not None:
        config.chunk_overlap_blocks = args.chunk_overlap_blocks
    if getattr(args, "concurrency", None) is not None:
        config.concurrency = args.concurrency
    actor_env = os.environ.get("CONSTRAINT_ACTOR_MODEL")
    critic_env = os.environ.get("CONSTRAINT_CRITIC_MODEL")
    base_env = os.environ.get("OPENAI_BASE_URL")
    if actor_env and not getattr(args, "actor_model", None):
        config.actor_model = actor_env
    if critic_env and not getattr(args, "critic_model", None):
        config.critic_model = critic_env
    if base_env and not getattr(args, "base_url", None):
        config.base_url = base_env
    return config


def _json_print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _add_preprocess_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("inputs", nargs="*", type=Path, help="MinerU Markdown files")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Artifact directory")
    parser.add_argument("-c", "--config", type=Path, help="JSON config file; relative input paths use the working directory")
    parser.add_argument("--chunk-max-chars", type=int, help="Override the chunk size from the configuration (default configuration: 12000)")
    parser.add_argument("--chunk-overlap-blocks", type=int, help="Override the context overlap from the configuration (default configuration: 2)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="constraint-pipeline",
        description="Direct actor-critic disease constraint extraction",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess", help="Parse and chunk Markdown text")
    _add_preprocess_options(preprocess)

    run = subparsers.add_parser("run", help="Run the actor-critic extraction")
    _add_preprocess_options(run)
    run.add_argument("--actor-model", help="Extraction model ID; overrides CONSTRAINT_ACTOR_MODEL and the configuration")
    run.add_argument("--critic-model", help="Review model ID; overrides CONSTRAINT_CRITIC_MODEL and the configuration")
    run.add_argument("--base-url", help="Service URL including /v1; overrides OPENAI_BASE_URL and the configuration")
    run.add_argument(
        "--concurrency",
        type=int,
        help="Maximum concurrent chunk/type extraction jobs (default: 6)",
    )

    compact = subparsers.add_parser(
        "compact",
        help="Build normalized compact artifacts from an existing run",
    )
    compact.add_argument("run_directory", type=Path)

    evaluate = subparsers.add_parser("evaluate", help="Compare predictions with adjudicated gold JSONL")
    evaluate.add_argument("--pred", type=Path, required=True)
    evaluate.add_argument("--gold", type=Path, required=True)
    evaluate.add_argument("--min-precision", type=float, default=0.0)
    evaluate.add_argument("--min-recall", type=float, default=0.0)
    evaluate.add_argument("--report", type=Path)
    export = subparsers.add_parser("export", help="Export atomic JSONL to grouped knowledge JSON")
    export.add_argument("--input", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)

    build = subparsers.add_parser("build", help="Build a Curasyn SQLite knowledge base")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--mapping", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "preprocess":
            config = _load_config(args)
            config.validate(require_models=False)
            inputs = args.inputs or [Path(p) for p in config.inputs if p.strip()]
            _, manifest = prepare_documents(inputs, args.output, config)
            _json_print(manifest)
            return 0
        if args.command == "run":
            config = _load_config(args)
            config.validate(require_models=True)
            inputs = args.inputs or [Path(p) for p in config.inputs if p.strip()]
            if not inputs:
                raise ValueError("Fill in inputs with Markdown paths or pass them on the command line")
            from .test_dictionary import configure_scope
            configure_scope(
                config.ehr_path, config.dictionary_path, config.hpo_path,
                config.hpo_vocabulary_path,
            )
            client = OpenAICompatibleClient(config, args.output / "calls")
            manifest = run_pipeline(
                input_paths=inputs,
                output_directory=args.output,
                config=config,
                client=client,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            _json_print(manifest)
            return 0 if manifest["status"] == "complete" else 2
        if args.command == "export":
            from .export import convert_file
            _json_print(convert_file(args.input, args.output))
            return 0
        if args.command == "build":
            from constraintKB.build_kb import build_database
            _json_print(build_database(args.source, args.mapping, args.output))
            return 0
        if args.command == "compact":
            run_directory = args.run_directory
            constraints = read_jsonl(run_directory / "constraints.jsonl")
            stats = write_compact_outputs(run_directory, constraints)
            manifest_path = run_directory / "manifest.json"
            if manifest_path.is_file():
                manifest = read_json(manifest_path)
                output = manifest.setdefault("output", {})
                output["compact_constraint_groups"] = stats["constraint_groups"]
                output["constraint_evidence_records"] = stats[
                    "unique_evidence_records"
                ]
                manifest["compaction"] = stats
                manifest.setdefault("guarantees", {})[
                    "compact_constraints_normalized"
                ] = True
                write_json(manifest_path, manifest)
            _json_print(stats)
            return 0
        if args.command == "evaluate":
            if not 0 <= args.min_precision <= 1 or not 0 <= args.min_recall <= 1:
                raise ValueError("Evaluation thresholds must be between 0 and 1")
            report = evaluate_constraints(
                read_jsonl(args.pred),
                read_jsonl(args.gold),
                min_precision=args.min_precision,
                min_recall=args.min_recall,
            )
            if args.report:
                write_json(args.report, report)
            _json_print(report)
            return 0 if report["passed_thresholds"] else 2
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 1
