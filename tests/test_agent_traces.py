import copy
import json

import pytest

from finautorubric.evaluation import grade
from finautorubric.evaluation.common.io import read_json, read_jsonl
from finautorubric.evaluation.common.judge import load_answers, response_for_judge
from finautorubric.evaluation.common.trace import RENDERING_POLICY, render_trace
from finautorubric.evaluation.ensemble import merge
from finautorubric.evaluation.public import evaluate, load_config
from .test_judge_panel import panel_command
from .test_public_evaluation import FakeJudge, command, inputs, write_rows


TRACE = [
    {"step": 1, "assistant_text": "Calculate the operating margin.",
     "tool_calls": [{"id": "call-1", "name": "calculator", "input": {"expression": "20 / 100 * 100"}}],
     "tool_results": [{"tool_use_id": "call-1", "content": "20", "is_error": False}]},
    {"step": 2, "assistant_text": "The calculation gives 20%."},
]


@pytest.mark.parametrize("wrapper", [None, "response", "payload"])
@pytest.mark.parametrize("outer_trace", [False, True])
def test_answer_and_trace_formats_preserve_sources_and_limitations(tmp_path, wrapper, outer_trace):
    response = {"answer": "20%", "trace": copy.deepcopy(TRACE), "sources": ["https://example.com/report"],
                "limitations": "Uses the supplied figures."}
    row = {"query_id": "FAR_example", "answer_id": "agent"}
    if wrapper:
        row[wrapper] = copy.deepcopy(response)
        if outer_trace:
            row["trace"] = row[wrapper].pop("trace")
    else:
        row.update(response)
    path = tmp_path / "answers.jsonl"
    write_rows(path, [row])
    _, answers = load_answers(f"answers={path}")
    loaded = answers["FAR_example"][0]["response"]
    assert loaded == response
    judged = response_for_judge(loaded)
    assert judged["sources"] == response["sources"]
    assert judged["limitations"] == response["limitations"]
    assert judged["answer"].startswith("AGENT'S FINAL ANSWER:\n20%\n\nAGENT'S TRACE")
    assert judged["answer"].index("=== step 1 ===") < judged["answer"].index("tool_call calculator")
    assert judged["answer"].index("tool_result: 20") < judged["answer"].index("=== step 2 ===")


@pytest.mark.parametrize("trace", [
    None, "raw trace", {}, ["step"], [{}], [{"messages": []}],
    [{"assistant_text": 1}], [{"assistant_text": "x", "step": True}],
    [{"assistant_text": "x", "step": -1}],
    [{"assistant_text": "x", "step": 2}, {"assistant_text": "y", "step": 1}],
    [{"tool_calls": {}}], [{"tool_calls": [None]}],
    [{"tool_calls": [{"name": "calculator", "input": {}}]}],
    [{"tool_calls": [{"id": "c", "name": "calculator", "input": "20/100"}]}],
    [{"tool_calls": [{"id": "c", "name": "calculator", "input": {"x": float("nan")}}]}],
    [{"tool_results": [{"tool_use_id": "c", "content": 20}]}],
    [{"tool_results": [{"tool_use_id": "c", "content": "20", "is_error": "false"}]}],
])
def test_malformed_trace_fails_before_artifacts_or_model_calls(inputs, tmp_path, monkeypatch, trace):
    _, answers, _, row = inputs
    write_rows(answers, [{"query_id": row["query_id"], "answer_id": "agent", "answer": "20%", "trace": trace}])
    monkeypatch.setattr(grade, "make_judge", lambda *a, **k: pytest.fail("opened a backend"))
    out = tmp_path / "bad"
    with pytest.raises(ValueError):
        evaluate(panel_command(inputs, out))
    assert not out.exists()


@pytest.mark.parametrize("fields", [
    {"answer": "20%", "trace": [], "response": {"answer": "other", "trace": []}},
    {"response": {"answer": "20%", "trace": []}, "trace": TRACE},
    {"response": {"answer": None, "trace": []}},
])
def test_missing_or_ambiguous_submission_is_rejected(inputs, tmp_path, monkeypatch, fields):
    _, answers, _, row = inputs
    write_rows(answers, [{"query_id": row["query_id"], "answer_id": "agent", **fields}])
    monkeypatch.setattr(grade, "make_judge", lambda *a, **k: pytest.fail("opened a backend"))
    out = tmp_path / "bad"
    with pytest.raises(ValueError):
        evaluate(panel_command(inputs, out))
    assert not out.exists()


@pytest.mark.parametrize("wrapper", ["answer", "response", "payload"])
def test_answer_without_trace_can_receive_full_score(inputs, tmp_path, monkeypatch, wrapper):
    _, answers, _, row = inputs
    answer = "The operating margin is 20%: 20 / 100 * 100 = 20%."
    write_rows(answers, [{"query_id": row["query_id"], "answer_id": "agent",
                         wrapper: answer if wrapper == "answer" else {"answer": answer}}])
    backends = {config["name"]: FakeJudge(passing_text="") for config in load_config()}
    monkeypatch.setattr(grade, "make_judge", lambda name, **kw: backends[name])
    out = tmp_path / "answer-only"
    assert evaluate(panel_command(inputs, out)) == 0
    assert read_json(out / "scores.json")["aggregate"][0]["mean_score"] == 100
    for name, backend in backends.items():
        frozen = read_json(out / "members" / name / "frozen/answers/answers/agent/FAR_example.json")
        assert frozen["response"] == {"answer": answer}
        assert "[trace not provided]" in frozen["candidate_response"]["answer"]
        assert "[trace not provided]" in backend.calls[0]["prompt"]
    monkeypatch.setattr(grade, "make_judge", lambda *a, **kw: pytest.fail("reopened completed run"))
    assert evaluate(panel_command(inputs, out, "--resume")) == 0


def test_rendering_caps_preserve_final_answer_and_trace_head_tail():
    trace = copy.deepcopy(TRACE)
    trace[0]["tool_calls"][0]["input"] = {"z": "한" * 2000, "a": 1}
    trace[0]["tool_results"][0].update(content="나" * 4000 + "RESULT_TAIL", is_error=True)
    original = copy.deepcopy(trace)
    body = render_trace(trace)
    args = json.dumps(trace[0]["tool_calls"][0]["input"], ensure_ascii=False, sort_keys=True)
    assert f"tool_call calculator({args[:1500]}...)" in body
    assert "tool_result [ERROR]: " + "나" * 4000 + "..." in body
    assert "RESULT_TAIL" not in body
    assert trace == original
    # Assistant text is only bounded by the whole-trace cap.
    trace.append({"assistant_text": "HEAD" + "x" * 75000 + "OMITTED_MIDDLE" + "y" * 75000 + "TAIL"})
    final = "FINAL_ANSWER_SENTINEL" * 10000
    judged = response_for_judge({"answer": final, "trace": trace})["answer"]
    assert judged.startswith("AGENT'S FINAL ANSWER:\n" + final)
    body = judged.split("AGENT'S TRACE (assistant text, tool calls, tool results):\n", 1)[1]
    marker = "\n\n... [trace truncated for length] ...\n\n"
    assert len(body) == 150000 + len(marker)
    assert body.startswith("=== step 1 ===") and body.endswith("TAIL")
    assert "OMITTED_MIDDLE" not in body


def test_argument_key_order_does_not_change_rendered_input():
    first, second = copy.deepcopy(TRACE), copy.deepcopy(TRACE)
    first[0]["tool_calls"][0]["input"] = {"b": 2, "a": 1}
    second[0]["tool_calls"][0]["input"] = {"a": 1, "b": 2}
    assert render_trace(first) == render_trace(second)


def test_trace_reaches_every_judge_and_changes_the_mock_score(inputs, tmp_path, monkeypatch):
    _, answers, _, row = inputs

    class TraceJudge(FakeJudge):
        async def generate(self, **kwargs):
            result = await super().generate(**kwargs)
            payload = json.loads(kwargs["prompt"].split("```json\n", 1)[1].rsplit("\n```", 1)[0])
            candidate = payload["candidate_response"]
            if 'tool_call calculator({"expression": "20 / 100 * 100"})' in candidate["answer"]:
                for verdict in result.output["verdicts"]:
                    verdict["pass"] = True
            return result

    backends = {config["name"]: TraceJudge() for config in load_config()}
    monkeypatch.setattr(grade, "make_judge", lambda name, **kw: backends[name])
    empty = tmp_path / "empty-trace"
    assert evaluate(panel_command(inputs, empty)) == 0
    assert read_json(empty / "scores.json")["aggregate"][0]["mean_score"] == 60
    write_rows(answers, [{"query_id": row["query_id"], "answer_id": "agent", "answer": "My answer", "trace": TRACE}])
    out = tmp_path / "with-trace"
    assert evaluate(panel_command(inputs, out, "--dry-run")) == 0
    for name in backends:
        frozen = read_json(out / "members" / name / "frozen/answers/answers/agent/FAR_example.json")
        assert frozen["response"] == {"answer": "My answer", "trace": TRACE}
        assert frozen["candidate_response"] == response_for_judge(frozen["response"])
    assert evaluate(panel_command(inputs, out, "--resume")) == 0
    assert read_json(out / "scores.json")["aggregate"][0]["mean_score"] == 100
    assert all(len(backend.calls) == 2 for backend in backends.values())
    prompts = [backend.calls[-1]["prompt"] for backend in backends.values()]
    assert len(set(prompts)) == 1
    manifest = read_json(out / "ensemble/manifest.json")
    assert manifest["answer_rendering"] == RENDERING_POLICY
    monkeypatch.setattr(grade, "make_judge", lambda *a, **kw: pytest.fail("reopened completed run"))
    assert evaluate(panel_command(inputs, out, "--resume")) == 0


@pytest.mark.parametrize("change", ["trace", "frozen_trace", "frozen_candidate", "policy"])
def test_trace_or_renderer_changes_cannot_resume(inputs, tmp_path, monkeypatch, change):
    out = tmp_path / "run"
    assert evaluate(command(inputs, out, "--dry-run")) == 0
    monkeypatch.setattr(grade, "make_judge", lambda *a, **kw: pytest.fail("opened backend"))
    if change == "trace":
        rows = read_jsonl(inputs[1])
        rows[0]["trace"] = TRACE
        write_rows(inputs[1], rows)
    elif change.startswith("frozen"):
        path = out / "frozen/answers/answers/agent/FAR_example.json"
        frozen = read_json(path)
        if change == "frozen_trace":
            frozen["response"]["trace"] = TRACE
        else:
            frozen["candidate_response"]["answer"] = "different"
        path.write_text(json.dumps(frozen))
    else:
        monkeypatch.setitem(RENDERING_POLICY, "version", "different-version")
    with pytest.raises(ValueError, match="changed"):
        evaluate(command(inputs, out, "--resume"))


def test_merge_rejects_different_rendering_policy(inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(grade, "make_judge", lambda *a, **kw: FakeJudge())
    out = tmp_path / "panel"
    assert evaluate(panel_command(inputs, out)) == 0
    members = [out / "members" / config["name"] for config in load_config()]
    path = members[0] / "manifest.json"
    manifest = read_json(path)
    manifest["answer_rendering"]["version"] = "different"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="rendering policy"):
        merge(members, tmp_path / "bad-merge")
