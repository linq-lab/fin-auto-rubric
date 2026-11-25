"""OpenRouter chat-completions judge backend (structured output over stdlib HTTP).

Implements `generate(stage=..., system_prompt=..., prompt=..., output_schema=...)`
for criterion judging. One request per stage: system prompt as the system message, stage input as the user
message, `response_format=json_schema` (strict) and `reasoning.effort`. Provider routing
(`data_collection`, `quantizations`, `require_parameters`, ...) comes from the judge spec
and is recorded in every trace. Each HTTP attempt waits at most `request_timeout_seconds`
(default 300 s), so a stalled connection is retried inside the stage budget instead of
consuming all of it. The API key is read from `OPENROUTER_API_KEY` (environment,
then the repo `.env`); its value never reaches traces, metadata, or error messages.

Trace layout per stage (next free attempt, so a failed attempt never blocks a resume)::

    {trace_dir}/{stage}/attempt_NN/prompt.txt, schema.json, request.json, response.json, metadata.json
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Optional

from .backends import AgentResult, BackendError, BackendTimeoutError
from .io import atomic_write_json, canonical_sha256

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY_NAME = "OPENROUTER_API_KEY"
RETRY_STATUS = (408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529)
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def load_api_key(name: str, env_file: Path) -> str:
    """Environment first, then a KEY=VALUE env file; the value itself is never logged."""

    value = os.environ.get(name, "")
    if not value and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):]
            key, sep, raw = line.partition("=")
            if sep and key.strip() == name:
                value = raw.strip().strip('"').strip("'")
    if not value:
        raise BackendError(f"{name} is not set (export it or put it in {env_file})")
    return value


class OpenRouterBackend:
    """One structured-output chat completion per stage through OpenRouter."""

    def __init__(
        self,
        *,
        model: str,
        reasoning_effort: str,
        timeout_seconds: float,
        trace_dir: Optional[Path],
        provider: Optional[Mapping[str, Any]] = None,
        max_tokens: int | None = 32768,
        max_retries: int = 4,
        request_timeout_seconds: float = 300.0,
        api_key: Optional[str] = None,
        env_file: Optional[Path] = None,
        max_web_search_calls: int = 0,
        transport=None,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = float(timeout_seconds)
        self.trace_dir = Path(trace_dir) if trace_dir is not None else None
        self.provider = dict(provider or {})
        self.max_tokens = int(max_tokens) if max_tokens is not None else None
        self.max_retries = int(max_retries)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.max_web_search_calls = max_web_search_calls
        self._key = api_key or load_api_key(API_KEY_NAME, env_file or Path.cwd() / ".env")
        self._transport = transport or self._post

    # -- HTTP -----------------------------------------------------------------

    def _post(self, body: bytes, timeout: float) -> tuple[int, str]:
        request = urllib.request.Request(CHAT_URL, data=body, headers={
            "Authorization": "Bearer " + self._key, "Content-Type": "application/json",
            "X-Title": "FinAutoRubric",
        })
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", errors="replace")

    def _redact(self, text: str) -> str:
        return text.replace(self._key, "<key>") if self._key else text

    # -- traces ---------------------------------------------------------------

    def _attempt_root(self, stage: str) -> Optional[Path]:
        if self.trace_dir is None:
            return None
        base = self.trace_dir / re.sub(r"[^a-zA-Z0-9_.-]+", "_", stage)
        base.mkdir(parents=True, exist_ok=True)
        number = 1
        while True:
            root = base / f"attempt_{number:02d}"
            try:
                root.mkdir()
                return root
            except FileExistsError:
                number += 1

    def _write(self, root: Optional[Path], name: str, payload: Any) -> None:
        if root is None:
            return
        if isinstance(payload, str):
            (root / name).write_text(self._redact(payload), encoding="utf-8")
        else:
            atomic_write_json(root / name, json.loads(self._redact(json.dumps(payload))))

    # -- generate -------------------------------------------------------------

    def request_body(self, system_prompt: str, prompt: str, output_schema: Mapping[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "judge_output", "strict": True, "schema": dict(output_schema)}},
            "reasoning": {"effort": self.reasoning_effort},
            "usage": {"include": True},
        }
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        if self.provider:
            body["provider"] = dict(self.provider)
        return body

    @staticmethod
    def parse_content(content: Any) -> dict[str, Any]:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("empty content")
        text = content.strip()
        fenced = _FENCE.match(text)
        if fenced:
            text = fenced.group(1)
        output = json.loads(text)
        if not isinstance(output, dict):
            raise ValueError("structured output must be an object")
        return output

    async def generate(self, *, stage: str, system_prompt: str, prompt: str, output_schema: Mapping[str, Any], allow_web: bool = False) -> AgentResult:
        if allow_web:
            raise BackendError("OpenRouter judge backend does not support web search (stage %s)" % stage)
        root = self._attempt_root(stage)
        body = self.request_body(system_prompt, prompt, output_schema)
        self._write(root, "prompt.txt", system_prompt.rstrip() + "\n\n<stage_input>\n" + prompt.rstrip() + "\n</stage_input>\n")
        self._write(root, "schema.json", dict(output_schema))
        self._write(root, "request.json", {k: v for k, v in body.items() if k != "messages"})
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        started = time.monotonic()
        errors: list[str] = []
        for attempt in range(1, self.max_retries + 1):
            remaining = self.timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise BackendTimeoutError(backend="OpenRouter", stage=stage, timeout_seconds=self.timeout_seconds)
            try:
                status, text = await asyncio.to_thread(self._transport, encoded, min(remaining, self.request_timeout_seconds))
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                errors.append(f"attempt {attempt}: transport {type(exc).__name__}: {self._redact(str(exc))[:200]}")
                await asyncio.sleep(min(60.0, 5.0 * attempt))
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = {"raw": text[:2000]}
            self._write(root, f"response_{attempt:02d}.json", payload)
            error = payload.get("error") if isinstance(payload, dict) else None
            if status != 200 or error:
                detail = self._redact(json.dumps(error or payload, ensure_ascii=False))[:400]
                errors.append(f"attempt {attempt}: HTTP {status}: {detail}")
                code = (error or {}).get("code") if isinstance(error, dict) else None
                if status in RETRY_STATUS or code in RETRY_STATUS:
                    await asyncio.sleep(min(60.0, 5.0 * attempt))
                    continue
                raise BackendError("OpenRouter stage failed (%s): %s" % (stage, "; ".join(errors)))
            choice = (payload.get("choices") or [{}])[0]
            try:
                output = self.parse_content((choice.get("message") or {}).get("content"))
            except (ValueError, json.JSONDecodeError) as exc:
                # Truncated/empty/non-JSON structured output: resample once more within the retry budget.
                errors.append(f"attempt {attempt}: invalid structured output ({exc}); finish={choice.get('finish_reason')}")
                continue
            elapsed = time.monotonic() - started
            usage = payload.get("usage") or {}
            metadata = {
                "backend": "openrouter",
                "model": self.model,
                "served_model": payload.get("model"),
                "provider_name": payload.get("provider"),
                "provider_routing": self.provider,
                "generation_id": payload.get("id"),
                "reasoning_effort": self.reasoning_effort,
                "allow_web": False,
                "web_search_calls": 0,
                "max_web_search_calls": self.max_web_search_calls,
                "timeout_seconds": self.timeout_seconds,
                "request_timeout_seconds": self.request_timeout_seconds,
                "elapsed_seconds": round(elapsed, 3),
                "http_attempts": attempt,
                "finish_reason": choice.get("finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                "cost_usd": usage.get("cost"),
                "prompt_sha256": canonical_sha256(body["messages"]),
                "schema_sha256": canonical_sha256(output_schema),
                "output_sha256": canonical_sha256(output),
            }
            self._write(root, "final.json", output)
            self._write(root, "metadata.json", metadata)
            return AgentResult(output=output, metadata=metadata)
        raise BackendError("OpenRouter stage failed after %d attempts (%s): %s" % (self.max_retries, stage, "; ".join(errors)[-1200:]))


__all__ = ["CHAT_URL", "OpenRouterBackend"]
