from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PipelineConfig:
    inputs: list[str] = field(default_factory=list)
    ehr_path: str = ""
    dictionary_path: str = ""
    hpo_path: str = ""
    hpo_vocabulary_path: str = ""
    actor_model: str = ""
    critic_model: str = ""
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    require_api_key: bool = True
    response_format: str = "json_schema"
    concurrency: int = 6
    max_rounds: int = 3
    critic_format_retries: int = 1
    request_retries: int = 3
    request_timeout_seconds: int = 180
    max_output_tokens: int = 12000
    actor_revision_request_retries: int = 1
    actor_revision_request_timeout_seconds: int = 360
    actor_revision_max_output_tokens: int = 10000
    token_parameter: str = "max_tokens"
    temperature: float | None = 0.0
    enable_thinking: bool | None = None
    chunk_max_chars: int = 12000
    chunk_overlap_blocks: int = 2

    @classmethod
    def from_file(cls, path: Path) -> "PipelineConfig":
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("Config root must be a JSON object")
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")
        config = cls(**raw)
        config.validate(require_models=False)
        return config

    def validate(self, *, require_models: bool = True) -> None:
        if not isinstance(self.inputs, list) or any(not isinstance(p, str) for p in self.inputs):
            raise ValueError("inputs must be a list of Markdown paths")
        string_fields = {
            "ehr_path": self.ehr_path,
            "dictionary_path": self.dictionary_path,
            "hpo_path": self.hpo_path,
            "hpo_vocabulary_path": self.hpo_vocabulary_path,
            "actor_model": self.actor_model,
            "critic_model": self.critic_model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "response_format": self.response_format,
            "token_parameter": self.token_parameter,
        }
        invalid_strings = sorted(
            name for name, value in string_fields.items() if not isinstance(value, str)
        )
        if invalid_strings:
            raise ValueError(f"Config fields must be strings: {invalid_strings}")
        if not isinstance(self.require_api_key, bool):
            raise ValueError("require_api_key must be boolean")
        integer_fields = {
            "concurrency": self.concurrency,
            "max_rounds": self.max_rounds,
            "critic_format_retries": self.critic_format_retries,
            "request_retries": self.request_retries,
            "request_timeout_seconds": self.request_timeout_seconds,
            "max_output_tokens": self.max_output_tokens,
            "actor_revision_request_retries": self.actor_revision_request_retries,
            "actor_revision_request_timeout_seconds": (
                self.actor_revision_request_timeout_seconds
            ),
            "actor_revision_max_output_tokens": self.actor_revision_max_output_tokens,
            "chunk_max_chars": self.chunk_max_chars,
            "chunk_overlap_blocks": self.chunk_overlap_blocks,
        }
        invalid_integers = sorted(
            name
            for name, value in integer_fields.items()
            if not isinstance(value, int) or isinstance(value, bool)
        )
        if invalid_integers:
            raise ValueError(f"Config fields must be integers: {invalid_integers}")
        if require_models and (
            not self.actor_model.strip() or not self.critic_model.strip()
        ):
            raise ValueError("actor_model and critic_model are required for extraction")
        if require_models and not self.hpo_vocabulary_path.strip():
            raise ValueError("hpo_vocabulary_path is required for extraction; use a fixed task vocabulary file")
        if self.response_format not in {"json_schema", "json_object", "prompt_only"}:
            raise ValueError("response_format must be json_schema, json_object, or prompt_only")
        if self.concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        if (
            self.critic_format_retries < 0
            or self.request_retries < 0
            or self.actor_revision_request_retries < 0
        ):
            raise ValueError("retry counts cannot be negative")
        if (
            self.request_timeout_seconds < 1
            or self.max_output_tokens < 1
            or self.actor_revision_request_timeout_seconds < 1
            or self.actor_revision_max_output_tokens < 1
        ):
            raise ValueError("request timeout and max output tokens must be positive")
        if self.token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("token_parameter must be max_tokens or max_completion_tokens")
        if self.temperature is not None:
            if not isinstance(self.temperature, (int, float)) or isinstance(
                self.temperature, bool
            ):
                raise ValueError("temperature must be numeric or null")
            if not 0 <= self.temperature <= 2:
                raise ValueError("temperature must be null or between 0 and 2")
        if self.enable_thinking is not None and not isinstance(self.enable_thinking, bool):
            raise ValueError("enable_thinking must be boolean or null")
        if self.chunk_max_chars < 1000:
            raise ValueError("chunk_max_chars must be at least 1000")
        if self.chunk_overlap_blocks < 0:
            raise ValueError("chunk_overlap_blocks cannot be negative")
        if not self.base_url.strip():
            raise ValueError("base_url cannot be empty")
        if not self.api_key_env.strip():
            raise ValueError("api_key_env cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
