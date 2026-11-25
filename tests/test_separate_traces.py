import copy
import json
from pathlib import Path
import re

import pytest

from finautorubric.cli import main
from finautorubric.evaluation import grade
from finautorubric.evaluation.common.io import file_sha256, read_json, read_jsonl
from finautorubric.evaluation.common.judge import load_answers
from finautorubric.evaluation.public import evaluate, load_config
from .test_agent_traces import TRACE
from .test_judge_panel import panel_command
from .test_public_evaluation import FakeJudge, inputs, write_rows


@pytest.fixture
def separate_inputs(inputs, tmp_path):
    row = read_jsonl(inputs[1])[0]
    row.pop("trace")
    write_rows(inputs[1], [row])
    traces = tmp_path / "traces.jsonl"
    write_rows(traces, [{"query_id": row["query_id"], "answer_id": row["answer_id"], "trace": copy.deepcopy(TRACE)}])
    return inputs, traces


@pytest.mark.parametrize("wrapper", ["answer", "response", "payload"])
def test_separate_traces_join_by_both_ids_not_line_order(tmp_path, wrapper):
    answers, traces = tmp_path / "answers.jsonl", tmp_path / "traces.jsonl"
    pairs = [("q1", "a"), ("q1", "b"), ("q2", "a"), ("q2", "b")]
    write_rows(answers, [{"query_id": q, "answer_id": a,
                         wrapper: f"{q}/{a}" if wrapper == "answer" else {"answer": f"{q}/{a}"}}
                        for q, a in pairs])
    write_rows(traces, [{"query_id": q, "answer_id": a, "trace": [{"assistant_text": f"trace for {q}/{a}"}]}
                       for q, a in reversed(pairs)])
    _, loaded = load_answers(f"answers={answers}", traces_path=traces)
    for q, rows in loaded.items():
        for row in rows:
            assert row["response"] == {"answer": f"{q}/{row['answer_id']}",
                                       "trace": [{"assistant_text": f"trace for {q}/{row['answer_id']}"}]}
            assert row["trace_source_path"] == str(traces)


@pytest.mark.parametrize("bad", ["duplicate", "extra", "wrong_system", "wrong_query",
                                 "missing_field", "malformed", "no_query_id", "no_answer_id",
                                 "bad_id", "inline", "nested_inline", "answer_id_missing"])
def test_invalid_pairing_fails_before_model_calls(separate_inputs, tmp_path, monkeypatch, bad):
    inputs, traces = separate_inputs
    rows = read_jsonl(traces)
    if bad == "duplicate": rows *= 2
    elif bad == "extra": rows.append({**rows[0], "answer_id": "other"})
    elif bad == "wrong_system": rows[0]["answer_id"] = "other"
    elif bad == "wrong_query": rows[0]["query_id"] = "unknown"
    elif bad == "missing_field": rows[0].pop("trace")
    elif bad == "malformed": rows[0]["trace"] = "not a step list"
    elif bad == "no_query_id": rows[0].pop("query_id")
    elif bad == "no_answer_id": rows[0].pop("answer_id")
    elif bad == "bad_id": rows[0]["answer_id"] = []
    else:
        answer = read_jsonl(inputs[1])[0]
        if bad == "inline": answer["trace"] = []
        elif bad == "nested_inline": answer["response"] = {"answer": answer.pop("answer"), "trace": []}
        else: answer.pop("answer_id")
        write_rows(inputs[1], [answer])
    write_rows(traces, rows)
    out = tmp_path / "invalid"
    monkeypatch.setattr(grade, "make_judge", lambda *a, **kw: pytest.fail("opened backend"))
    with pytest.raises(ValueError):
        evaluate(panel_command(inputs, out, "--traces", str(traces)))
    assert not out.exists()


@pytest.mark.parametrize("include_trace", [False, True])
def test_trace_file_can_cover_only_some_or_none_of_the_answers(separate_inputs, tmp_path, monkeypatch, include_trace):
    inputs, traces = separate_inputs
    answer = read_jsonl(inputs[1])[0]
    write_rows(inputs[1], [answer, {**answer, "answer_id": "other"}])
    if not include_trace:
        write_rows(traces, [])
    monkeypatch.setattr(grade, "make_judge", lambda *a, **kw: FakeJudge())
    out = tmp_path / "optional-traces"
    assert evaluate(panel_command(inputs, out, "--traces", str(traces))) == 0
    for config in load_config():
        for answer_id in ("agent", "other"):
            frozen = read_json(out / "members" / config["name"] / f"frozen/answers/answers/{answer_id}/FAR_example.json")
            if include_trace and answer_id == "agent":
                assert frozen["response"]["trace"] == TRACE
                assert frozen["trace_source_path"] == str(traces)
            else:
                assert "trace" not in frozen["response"]
                assert "trace_source_path" not in frozen
                assert "[trace not provided]" in frozen["candidate_response"]["answer"]
    assert read_json(out / "scores.json")["complete"]


def test_separate_trace_reaches_panel_and_is_checked_on_resume(separate_inputs, tmp_path, monkeypatch):
    inputs, traces = separate_inputs
    out = tmp_path / "panel"
    args = panel_command(inputs, out, "--traces", str(traces))
    backends = {config["name"]: FakeJudge() for config in load_config()}
    monkeypatch.setattr(grade, "make_judge", lambda name, **kw: backends[name])
    assert evaluate([*args, "--dry-run"]) == 0
    assert all(not backend.calls for backend in backends.values())
    assert read_json(out / "request.json")["traces"] == file_sha256(traces)
    for name in backends:
        member = out / "members" / name
        frozen = read_json(member / "frozen/answers/answers/agent/FAR_example.json")
        assert frozen["response"]["trace"] == TRACE
        assert frozen["trace_source_path"] == str(traces)
        assert read_json(member / "request.json")["traces"] == file_sha256(traces)
    assert evaluate([*args, "--resume"]) == 0
    for backend in backends.values():
        payload = json.loads(backend.calls[0]["prompt"].split("```json\n", 1)[1].rsplit("\n```", 1)[0])
        assert "tool_call calculator" in payload["candidate_response"]["answer"]
        assert "tool_result: 20" in payload["candidate_response"]["answer"]
    monkeypatch.setattr(grade, "make_judge", lambda *a, **kw: pytest.fail("reopened backend"))
    assert evaluate([*args, "--resume"]) == 0
    rows = read_jsonl(traces)
    rows[0]["trace"][0]["assistant_text"] = "Changed trace"
    write_rows(traces, rows)
    with pytest.raises(ValueError, match="changed"):
        evaluate([*args, "--resume"])


@pytest.mark.parametrize("name", ["missing.jsonl", "traces.txt"])
def test_trace_flag_requires_a_jsonl_file(separate_inputs, tmp_path, name):
    inputs, traces = separate_inputs
    bad = tmp_path / name
    if name.endswith(".txt"):
        bad.write_bytes(traces.read_bytes())
    with pytest.raises(ValueError, match="--traces must be a JSONL file"):
        evaluate(panel_command(inputs, tmp_path / "bad", "--traces", str(bad), "--dry-run"))


@pytest.mark.parametrize("include_trace", [False, True])
def test_documented_separate_examples_prepare_without_model_calls(tmp_path, monkeypatch, include_trace):
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    for title, filename in [("**Answers JSONL", "answers.jsonl"), ("**Traces JSONL", "traces.jsonl")]:
        section = readme.split(title, 1)[1]
        documented = json.loads(re.search(r"```json\n(.*?)\n```", section, re.S).group(1))
        assert documented == read_jsonl(root / "examples" / filename)[0]
    monkeypatch.setattr(grade, "make_judge", lambda *a, **kw: pytest.fail("opened backend"))
    assert main(["evaluate", "--dataset", str(root / "examples/dataset.jsonl"),
                 "--answers", str(root / "examples/answers.jsonl"),
                 *(["--traces", str(root / "examples/traces.jsonl")] if include_trace else []),
                 "--output-dir", str(tmp_path / "example-run"), "--dry-run"]) == 0
