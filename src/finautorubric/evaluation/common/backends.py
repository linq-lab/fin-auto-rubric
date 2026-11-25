"""Judge client types and deterministic grading replays for offline tests."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Mapping, Protocol

from .io import canonical_sha256


class BackendError(RuntimeError):
    """A technical model-execution failure; never translated to unmatched."""


class BackendTimeoutError(BackendError):
    """A stage exceeded its configured wall-clock model-call timeout."""

    def __init__(self, *, backend: str, stage: str, timeout_seconds: float) -> None:
        self.backend = backend
        self.stage = stage
        self.timeout_seconds = timeout_seconds
        super().__init__(
            "%s stage timed out after %.1fs: %s"
            % (backend, timeout_seconds, stage)
        )


@dataclass
class AgentResult:
    output: Dict[str, Any]
    metadata: Dict[str, Any]


class AgentBackend(Protocol):
    async def generate(
        self,
        *,
        stage: str,
        system_prompt: str,
        prompt: str,
        output_schema: Mapping[str, Any],
        allow_web: bool = False,
    ) -> AgentResult:
        ...


class ReplayBackend:
    """Deterministic stage-keyed backend used by unit/integration tests."""

    def __init__(self, responses: Mapping[str, Any]) -> None:
        self._responses: Dict[str, Deque[Any]] = {}
        for stage, response in responses.items():
            values = response if isinstance(response, list) else [response]
            self._responses[stage] = deque(copy.deepcopy(values))
        self.calls: List[Dict[str, Any]] = []

    async def generate(
        self,
        *,
        stage: str,
        system_prompt: str,
        prompt: str,
        output_schema: Mapping[str, Any],
        allow_web: bool = False,
    ) -> AgentResult:
        self.calls.append(
            {
                "stage": stage,
                "system_prompt": system_prompt,
                "prompt": prompt,
                "output_schema": copy.deepcopy(dict(output_schema)),
                "allow_web": allow_web,
            }
        )
        queue = self._responses.get(stage)
        if not queue:
            raise BackendError("ReplayBackend has no response for stage=%s" % stage)
        response = queue.popleft()
        if isinstance(response, BaseException):
            raise response
        if not isinstance(response, dict):
            raise BackendError("ReplayBackend response must be an object")
        return AgentResult(
            output=copy.deepcopy(response),
            metadata={
                "backend": "replay",
                "model": "deterministic-fixture",
                "reasoning_effort": "none",
                "allow_web": allow_web,
                "output_sha256": canonical_sha256(response),
            },
        )
