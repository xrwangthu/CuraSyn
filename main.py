"""Main entrypoint for the complete Curasyn generation and curation pipeline."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from src.runner import run_curasyn
from src.schemas import SUPPORTED_DATASETS


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def apply_environment_overrides(
    config: dict[str, Any],
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    values = os.environ if environ is None else environ
    generation = config.setdefault("generation", {})
    api = generation.setdefault("api", {})

    base_url = values.get("CURASYN_LLM_BASE_URL")
    if base_url:
        api["base_url"] = base_url.rstrip("/")
    api_key = values.get("CURASYN_LLM_API_KEY")
    if api_key:
        api["api_key"] = api_key
    model = values.get("CURASYN_LLM_MODEL")
    if model:
        generation["model"] = model
    return config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Run from the repository root. Omitted overrides use the configuration "
            "file. Model and endpoint precedence: file < CURASYN_LLM environment "
            "variables < command-line options. Set input/output paths in config.yaml."
        ),
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"),
                        help="YAML configuration file (default: config.yaml)")
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS,
                        help="Dataset whose prepared files and support settings to use")
    parser.add_argument("--data-seed", type=int,
                        help="Generation seed when run.seed is null; does not change the fixed data split")
    parser.add_argument("--n-samples", type=int,
                        help="Final accepted record count; overrides generation.n_samples")
    parser.add_argument(
        "--serving",
        choices=["vllm", "together", "azure_openai", "openai_compatible"],
        help="Serving adapter; overrides generation.serving",
    )
    parser.add_argument("--model", help="Served model ID; overrides generation.model and CURASYN_LLM_MODEL")
    parser.add_argument("--artifact-tag", help="Output filename suffix; overrides run.artifact_tag")
    parser.add_argument(
        "--disease-weight-mode",
        choices=["empirical", "uniform", "custom"],
        help="Disease sampling weights; provide the custom mapping in the configuration file",
    )
    parser.add_argument("--max-exemplars", type=int, help="Maximum examples per prompt; overrides retrieval.max_exemplars")
    parser.add_argument("--theta", type=int, help="Minimum cohort size for residual retrieval; overrides retrieval.theta")
    parser.add_argument(
        "--retrieval-missing-policy",
        choices=["error", "median"],
        help="Numerical missing-value policy for retrieval",
    )
    parser.add_argument("--knowledge-source", type=Path, help="Static SQLite knowledge base; overrides knowledge.source")
    parser.add_argument("--standardized-confidence-threshold", type=float, help="Override the standardized curation confidence threshold")
    parser.add_argument("--standardized-uncertainty-threshold", type=float, help="Override the standardized curation uncertainty threshold")
    parser.add_argument(
        "--curation-missing-policy",
        choices=["error", "median"],
        help="Numerical missing-value policy for the curation encoder",
    )
    return parser.parse_args(argv)


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset:
        config["data"]["dataset"] = args.dataset
    if args.data_seed is not None:
        config["data"]["data_seed"] = args.data_seed
    if args.n_samples is not None:
        config["generation"]["n_samples"] = args.n_samples
    if args.serving:
        config["generation"]["serving"] = args.serving
    if args.model:
        config["generation"]["model"] = args.model
    if args.artifact_tag:
        config["run"]["artifact_tag"] = args.artifact_tag
    if getattr(args, "disease_weight_mode", None):
        config["generation"]["disease_weights"]["mode"] = (
            args.disease_weight_mode
        )
    if args.max_exemplars is not None:
        config["retrieval"]["max_exemplars"] = args.max_exemplars
    if args.theta is not None:
        config["retrieval"]["theta"] = args.theta
    if args.retrieval_missing_policy:
        config["retrieval"]["numerical_missing_policy"] = (
            args.retrieval_missing_policy
        )
    if getattr(args, "knowledge_source", None):
        config["knowledge"]["source"] = str(args.knowledge_source)
    if args.standardized_confidence_threshold is not None:
        config["curation"]["standardized_confidence_threshold"] = (
            args.standardized_confidence_threshold
        )
    if args.standardized_uncertainty_threshold is not None:
        config["curation"]["standardized_uncertainty_threshold"] = (
            args.standardized_uncertainty_threshold
        )
    if args.curation_missing_policy:
        config["curation"]["encoder"]["numerical_missing_policy"] = (
            args.curation_missing_policy
        )
    return config


def prepare_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build the effective main-pipeline configuration for one CLI invocation."""
    config = apply_environment_overrides(load_config(args.config))
    return apply_overrides(config, args)


def run_from_args(
    args: argparse.Namespace,
    *,
    config_transform: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the main pipeline with optional configuration customization."""
    config = prepare_config(args)
    if config_transform is not None:
        config_transform(config)
    return run_curasyn(config)


def main() -> None:
    try:
        args = parse_args()
        run_from_args(args)
    except Exception as error:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{timestamp}] [ERROR] {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
