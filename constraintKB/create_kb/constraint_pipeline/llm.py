from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from .config import PipelineConfig
from .io_utils import read_json, write_json


class LLMError(RuntimeError):
    pass


class LLMResponseError(LLMError):
    """The endpoint replied, but its content was not a usable JSON object."""


class CompletionClient(Protocol):
    def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        schema_name: str,
        call_id: str,
        request_timeout_seconds: int | None = None,
        request_retries: int | None = None,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]: ...


class OpenAICompatibleClient:
    def __init__(self, config: PipelineConfig, call_directory: Path) -> None:
        self.config = config
        self.call_directory = call_directory
        api_key = os.environ.get(config.api_key_env, "")
        if config.require_api_key and not api_key:
            raise LLMError(
                f"Environment variable {config.api_key_env} is unset; "
                "set it or use require_api_key=false for a trusted local endpoint"
            )
        self.api_key = api_key if config.require_api_key else ""

    def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        schema_name: str,
        call_id: str,
        request_timeout_seconds: int | None = None,
        request_retries: int | None = None,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        if self.config.response_format != "json_schema":
            schema_json = json.dumps(
                schema,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            user_prompt = f"""{user_prompt}

## Output Schema

The final JSON object must satisfy this schema exactly:

<json_schema>
{schema_json}
</json_schema>

## Final Answer Contract

Return exactly one JSON object that satisfies the schema. Return no commentary, Markdown, or
text outside that JSON object."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        resolved_max_output_tokens = (
            self.config.max_output_tokens
            if max_output_tokens is None
            else max_output_tokens
        )
        resolved_timeout = (
            self.config.request_timeout_seconds
            if request_timeout_seconds is None
            else request_timeout_seconds
        )
        resolved_retries = (
            self.config.request_retries
            if request_retries is None
            else request_retries
        )
        if (
            resolved_max_output_tokens < 1
            or resolved_timeout < 1
            or resolved_retries < 0
        ):
            raise LLMError("Per-call token, timeout, and retry overrides are invalid")

        payload[self.config.token_parameter] = resolved_max_output_tokens
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.enable_thinking is not None:
            payload["enable_thinking"] = self.config.enable_thinking
        if self.config.response_format == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            }
        elif self.config.response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}

        cache_material = {
            "endpoint": self._endpoint(),
            "payload": payload,
            "call_id": call_id,
        }
        request_hash = hashlib.sha256(
            json.dumps(cache_material, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cache_path = self.call_directory / f"{request_hash}.json"
        if cache_path.exists():
            cached = read_json(cache_path)
            if cached.get("request_hash") != request_hash:
                raise LLMError(f"Invalid call cache: {cache_path}")
            if "parse_error" in cached:
                raise LLMResponseError(
                    f"Cached LLM response was invalid: {cached['parse_error']}"
                )
            if "parsed_response" not in cached:
                raise LLMError(f"Invalid call cache: {cache_path}")
            parsed = cached["parsed_response"]
            if not isinstance(parsed, dict):
                raise LLMError(f"Cached LLM response is not an object: {cache_path}")
            return parsed

        response_json = self._request(
            payload,
            timeout_seconds=resolved_timeout,
            request_retries=resolved_retries,
        )
        try:
            content = self._message_content(response_json)
            parsed = self._parse_json_object(content)
        except LLMResponseError as exc:
            write_json(
                cache_path,
                {
                    "call_id": call_id,
                    "request_hash": request_hash,
                    "request": cache_material,
                    "raw_response": response_json,
                    "parse_error": str(exc),
                },
            )
            raise
        write_json(
            cache_path,
            {
                "call_id": call_id,
                "request_hash": request_hash,
                "request": cache_material,
                "raw_response": response_json,
                "parsed_response": parsed,
            },
        )
        return parsed

    def _endpoint(self) -> str:
        base = self.config.base_url.rstrip("/")
        return base if base.endswith("/chat/completions") else f"{base}/chat/completions"

    def _request(
        self,
        payload: dict[str, Any],
        *,
        timeout_seconds: int,
        request_retries: int,
    ) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error: Exception | None = None
        for attempt in range(request_retries + 1):
            request = urllib.request.Request(
                self._endpoint(), data=body, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=timeout_seconds
                ) as response:
                    value = json.loads(response.read().decode("utf-8"))
                if not isinstance(value, dict):
                    raise LLMError("LLM endpoint returned a non-object JSON response")
                return value
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:2000]
                last_error = LLMError(f"HTTP {exc.code} from LLM endpoint: {detail}")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, LLMError) as exc:
                last_error = exc
            if attempt < request_retries:
                time.sleep(min(2**attempt, 8))
        raise LLMError(f"LLM request failed after retries: {last_error}") from last_error

    @staticmethod
    def _message_content(response: dict[str, Any]) -> str:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError("LLM response has no choices[0].message.content") from exc
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            if parts:
                return "".join(parts)
        raise LLMResponseError("LLM message content is not textual JSON")

    @staticmethod
    def _parse_json_object(content: str) -> dict[str, Any]:
        stripped = content.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            stripped = fenced.group(1)
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"LLM returned invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise LLMResponseError("LLM returned JSON that is not an object")
        return value
