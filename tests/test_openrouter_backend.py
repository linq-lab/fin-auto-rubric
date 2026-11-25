"""Offline tests for the OpenRouter judge backend (fake transport, no network).

    python -m pytest tests/test_openrouter_backend.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from finautorubric.evaluation.common.backends import BackendError  # noqa: E402
from finautorubric.evaluation.common.openrouter import OpenRouterBackend  # noqa: E402

KEY = "sk-or-test-SECRET"
SCHEMA = {"type": "object", "properties": {"verdicts": {"type": "array"}}, "required": ["verdicts"], "additionalProperties": False}
OK = {"verdicts": [{"criterion_id": "C001", "pass": True, "rationale": "matches"}]}


def completion(content: str, *, provider: str = "DeepInfra") -> tuple[int, str]:
    return 200, json.dumps({
        "id": "gen-1", "model": "deepseek/deepseek-v4-flash-20260731", "provider": provider,
        "choices": [{"finish_reason": "stop", "message": {"content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.0001,
                  "completion_tokens_details": {"reasoning_tokens": 30}, "prompt_tokens_details": {"cached_tokens": 0}},
    })


class FakeTransport:
    def __init__(self, replies):
        self.replies = list(replies)
        self.bodies = []

    def __call__(self, body: bytes, timeout: float):
        self.bodies.append(json.loads(body))
        return self.replies.pop(0)


async def _no_sleep(_seconds):
    return None


class OpenRouterBackendTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def backend(self, transport) -> OpenRouterBackend:
        return OpenRouterBackend(model="deepseek/deepseek-v4-flash-0731", reasoning_effort="medium", timeout_seconds=30,
                                 trace_dir=self.tmp, provider={"data_collection": "deny", "quantizations": ["fp8", "bf16"]},
                                 api_key=KEY, transport=transport)

    @staticmethod
    def run_stage(backend, **kwargs):
        return asyncio.run(backend.generate(stage="grade_x_b001", system_prompt="SYS", prompt="USER", output_schema=SCHEMA, **kwargs))

    def test_request_shape_metadata_and_traces(self):
        fake = FakeTransport([completion(json.dumps(OK))])
        result = self.run_stage(self.backend(fake))
        body = fake.bodies[0]
        self.assertEqual(body["messages"], [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}])
        self.assertEqual(body["reasoning"], {"effort": "medium"})
        self.assertEqual(body["provider"], {"data_collection": "deny", "quantizations": ["fp8", "bf16"]})
        self.assertTrue(body["response_format"]["json_schema"]["strict"])
        self.assertEqual(body["response_format"]["json_schema"]["schema"], SCHEMA)
        self.assertEqual(result.output, OK)
        meta = result.metadata
        self.assertEqual((meta["backend"], meta["provider_name"], meta["reasoning_tokens"], meta["cost_usd"], meta["http_attempts"]),
                         ("openrouter", "DeepInfra", 30, 0.0001, 1))
        root = self.tmp / "grade_x_b001" / "attempt_01"
        self.assertTrue({"prompt.txt", "schema.json", "request.json", "response_01.json", "final.json", "metadata.json"} <= {p.name for p in root.iterdir()})
        self.assertFalse(any(KEY in p.read_text() for p in root.iterdir()))

    def test_explicit_none_omits_completion_token_limit(self):
        backend = OpenRouterBackend(model="m", reasoning_effort="medium", timeout_seconds=30,
                                    trace_dir=None, max_tokens=None, api_key=KEY)
        body = backend.request_body("SYS", "USER", SCHEMA)
        self.assertEqual(body["reasoning"], {"effort": "medium"})
        self.assertNotIn("max_tokens", body)
        self.assertNotIn("max_completion_tokens", body)
        self.assertEqual(self.backend(FakeTransport([])).request_body("SYS", "USER", SCHEMA)["max_tokens"], 32768)

    def test_retries_rate_limit_then_succeeds(self):
        fake = FakeTransport([(429, json.dumps({"error": {"code": 429, "message": "rate limited"}})), completion(json.dumps(OK))])
        with patch("finautorubric.evaluation.common.openrouter.asyncio.sleep", _no_sleep):
            result = self.run_stage(self.backend(fake))
        self.assertEqual(result.metadata["http_attempts"], 2)
        self.assertEqual(len(fake.bodies), 2)

    def test_fenced_json_and_invalid_output_resample(self):
        fake = FakeTransport([completion(""), completion("```json\n" + json.dumps(OK) + "\n```")])
        result = self.run_stage(self.backend(fake))
        self.assertEqual(result.output, OK)
        self.assertEqual(result.metadata["http_attempts"], 2)

    def test_non_retryable_error_raises_without_key(self):
        fake = FakeTransport([(404, json.dumps({"error": {"code": 404, "message": "No endpoints found matching your data policy " + KEY}}))])
        with self.assertRaises(BackendError) as ctx:
            self.run_stage(self.backend(fake))
        self.assertIn("data policy", str(ctx.exception))
        self.assertNotIn(KEY, str(ctx.exception))
        self.assertFalse(any(KEY in p.read_text() for p in self.tmp.rglob("*.json")))

    def test_web_disallowed_and_attempt_dirs_never_collide(self):
        with self.assertRaises(BackendError):
            self.run_stage(self.backend(FakeTransport([])), allow_web=True)
        backend = self.backend(FakeTransport([completion(json.dumps(OK)), completion(json.dumps(OK))]))
        self.run_stage(backend)
        self.run_stage(backend)
        self.assertEqual(sorted(p.name for p in (self.tmp / "grade_x_b001").iterdir()), ["attempt_01", "attempt_02"])

    def test_stalled_request_is_retried_within_stage_budget(self):
        timeouts, replies = [], [TimeoutError("read timed out"), completion(json.dumps(OK))]

        def transport(body, timeout):
            timeouts.append(timeout)
            reply = replies.pop(0)
            if isinstance(reply, BaseException):
                raise reply
            return reply

        backend = OpenRouterBackend(model="m", reasoning_effort="medium", timeout_seconds=1200, trace_dir=self.tmp,
                                    api_key=KEY, transport=transport, request_timeout_seconds=300)
        with patch("finautorubric.evaluation.common.openrouter.asyncio.sleep", _no_sleep):
            result = self.run_stage(backend)
        self.assertEqual(result.metadata["http_attempts"], 2)
        self.assertEqual(timeouts[0], 300)
        self.assertEqual(result.metadata["request_timeout_seconds"], 300)


class RestoreBatchOrderTest(unittest.TestCase):
    def test_reorders_complete_sets_only(self):
        from finautorubric.evaluation import grade as module
        output = {"verdicts": [{"criterion_id": "C002", "pass": True, "rationale": "a"}, {"criterion_id": "C001", "pass": False, "rationale": "b"}]}
        self.assertTrue(module.restore_batch_order(output, ["C001", "C002"]))
        self.assertEqual([v["criterion_id"] for v in output["verdicts"]], ["C001", "C002"])
        self.assertFalse(module.restore_batch_order({"verdicts": [{"criterion_id": "C001"}]}, ["C001", "C002"]))
        self.assertFalse(module.restore_batch_order({"verdicts": [{"criterion_id": "C001"}, {"criterion_id": "C001"}]}, ["C001", "C002"]))


if __name__ == "__main__":
    unittest.main()
