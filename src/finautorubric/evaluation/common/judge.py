"""Judge backends, evaluator schema, and rubric/answer source loaders."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

from . import schemas as S
from .openrouter import OpenRouterBackend
from .io import read_json, read_jsonl
from .rubric import SCHEMA_VERSION, criterion_rows, validate_record
from .trace import render_answer, validate_response

# Populated from the explicit public judge configuration before evaluation.
JUDGE_SPECS: dict[str, dict[str, Any]] = {}
DEFAULT_JUDGE = "judge"


def make_judge(judge_key: str, *, timeout_seconds: float, trace_dir: Path) -> OpenRouterBackend:
    if judge_key not in JUDGE_SPECS:
        raise ValueError(f"judge {judge_key!r} has no configuration")
    spec = JUDGE_SPECS[judge_key]
    return OpenRouterBackend(model=spec["model"], reasoning_effort=spec["reasoning_effort"], provider=spec.get("provider"),
                             timeout_seconds=timeout_seconds, trace_dir=trace_dir,
                             request_timeout_seconds=spec.get("request_timeout_seconds", 300.0),
                             max_tokens=spec.get("max_tokens", 32768), max_retries=spec.get("max_retries", 4))


def evaluator_schema(ids: list[str]) -> dict[str, Any]:
    """Identical to run_codex_sessions_v6_5.evaluator_schema."""

    evidence = S.obj({"claim": S.string(), "source_url": S.string(), "source_date": S.string(), "status": S.string(enum=["verified", "unverified", "contradicted"])})
    verdict = S.obj({"criterion_id": S.string(enum=ids), "pass": S.BOOLEAN, "rationale": S.string(min_length=1), "evidence": S.arr(evidence)})
    return S.obj({"verdicts": S.arr(verdict, min_items=len(ids), max_items=len(ids))})


def validate_verdicts(output: Mapping[str, Any], ids: list[str]) -> None:
    verdicts = output.get("verdicts") or []
    actual = [item.get("criterion_id") for item in verdicts]
    if actual != ids:
        raise ValueError(f"evaluator IDs/order mismatch: expected {ids}, got {actual}")
    if any(not isinstance(item.get("pass"), bool) for item in verdicts):
        raise ValueError("non-boolean evaluator verdict")


# ---------------------------------------------------------------------------
# rubric sources
# ---------------------------------------------------------------------------

def _legacy_to_record(row: Mapping[str, Any], *, label: str, weighting: str | None) -> dict[str, Any]:
    """Adapt `{query_id, criteria|rubric_public|rubric:[{rubric_text|text, points|weight}]}`."""

    raw = row.get("criteria") or row.get("rubric_public") or row.get("rubric")
    if isinstance(raw, dict):
        raw = raw.get("criteria") or raw.get("rubric")
    if not isinstance(raw, list):
        raise ValueError("cannot find a criteria list for %s in source %s" % (row.get("query_id"), label))
    rows = []
    for item in raw:
        text = item.get("rubric_text") or item.get("text") or item.get("requirement") or item.get("criterion")
        weight = item.get("points", item.get("weight", 1.0))
        rows.append({"rubric_text": text, "weight": float(weight), "meta": {"original_criterion_id": item.get("criterion_id")}})
    chosen = weighting or ("points" if any(r["weight"] != 1.0 for r in rows) else "uniform")
    record = {
        "schema_version": SCHEMA_VERSION, "method": label, "weighting": chosen, "query_id": row["query_id"], "benchmark": row.get("benchmark"),
        "information_cutoff": None, "criteria": criterion_rows(rows), "generation": {"adapted_from": "legacy_rows"}, "intermediate": {}, "prompt_hashes": {},
    }
    for normalized, original in zip(record["criteria"], raw):
        if original.get("criterion_id"):
            normalized["criterion_id"] = original["criterion_id"]
        if original.get("rubric_text"):
            normalized["rubric_text"] = original["rubric_text"]
    validate_record(record)
    return record


def load_rubric_source(spec: str, *, weighting: str | None = None) -> tuple[str, dict[str, dict[str, Any]]]:
    """`label=path[:method]` → (label, {query_id: baseline.rubric/v1 record}).

    * a baseline run directory (`items/{qid}/{method}/rubric.json`), `:method`
      defaults to the label;
    * a jsonl of canonical records, or of legacy `{query_id, criteria|rubric_public|rubric}` rows;
    * a directory of `{qid}.json` canonical records.
    """

    label, _, target = spec.partition("=")
    if not label or not target:
        raise ValueError("rubric source must be LABEL=PATH[:METHOD]: %r" % spec)
    path_text, _, method = target.partition(":")
    path = Path(path_text).expanduser().resolve()
    records: dict[str, dict[str, Any]] = {}
    if path.is_dir() and (path / "items").is_dir():
        method = method or label
        for rubric in sorted((path / "items").glob(f"*/{method}/rubric.json")):
            record = read_json(rubric)
            validate_record(record)
            records[record["query_id"]] = record
    elif path.is_dir():
        for file in sorted(path.glob("*.json")):
            record = read_json(file)
            validate_record(record)
            records[record["query_id"]] = record
    elif path.suffix == ".jsonl":
        for row in read_jsonl(path):
            if row.get("schema_version") == SCHEMA_VERSION:
                validate_record(row)
                records[row["query_id"]] = row
            else:
                record = _legacy_to_record(row, label=label, weighting=weighting)
                records[record["query_id"]] = record
    else:
        raise ValueError("unsupported rubric source path: %s" % path)
    contaminated = [qid for qid, rec in records.items() if rec.get("generation", {}).get("generation_status") == "contaminated"]
    for qid in contaminated:
        records.pop(qid)
    if contaminated:
        print(f"rubric source {label}: skipping {len(contaminated)} quarantined (contaminated) record(s)", file=sys.stderr, flush=True)
    if not records:
        raise ValueError("rubric source %s is empty" % spec)
    return label, records


# ---------------------------------------------------------------------------
# answers
# ---------------------------------------------------------------------------

def _response_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = [key for key in ("payload", "response", "answer") if key in row]
    if len(fields) != 1:
        raise ValueError("supply exactly one of answer, response, or payload")
    value = row[fields[0]]
    if isinstance(value, str):
        response = {"answer": value}
    elif isinstance(value, dict):
        response = dict(value)
    else:
        raise ValueError("answer must be text, or response/payload must be an object")
    for key in ("trace", "sources", "limitations"):
        if key in row:
            if key in response:
                raise ValueError(f"supply {key} only once, at the top level or inside response/payload")
            response[key] = row[key]
    validate_response(response)
    return response


def _submission_key(row: Mapping[str, Any]) -> tuple[str, str]:
    for field in ("query_id", "answer_id"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"Separate answer/trace files require a nonempty string {field} on every row")
    return row["query_id"], row["answer_id"]


def load_answers(spec: str, *, traces_path: Path | None = None) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    """`label=path` → (label, {query_id: [{answer_id, response}]}).

    * directory of `{query_id}.json` with a `payload` (final_bench style): one
      answer per query, `answer_id = label`;
    * jsonl rows `{query_id, answer_id, answer}` optionally paired with a separate
      traces JSONL by `(query_id, answer_id)`, or containing an inline `trace`;
    * structured `response`/`payload` is also supported.
    """

    label, _, path_text = spec.partition("=")
    if not label or not path_text:
        raise ValueError("answers must be LABEL=PATH: %r" % spec)
    path = Path(path_text).expanduser().resolve()
    traces = None
    used_traces = set()
    if traces_path is not None:
        traces_path = traces_path.expanduser().resolve()
        if path.suffix != ".jsonl" or path.is_dir():
            raise ValueError("Separate traces require an answers JSONL file")
        traces = {}
        for row in read_jsonl(traces_path):
            key = _submission_key(row)
            if key in traces:
                raise ValueError(f"Duplicate trace for query_id/answer_id: {key}")
            if "trace" not in row:
                raise ValueError(f"Missing trace field for query_id/answer_id: {key}")
            traces[key] = row["trace"]
    answers: dict[str, list[dict[str, Any]]] = {}
    if path.is_dir():
        for file in sorted(path.glob("*.json")):
            row = read_json(file)
            payload = _response_from_row(row)
            answers.setdefault(row.get("query_id", file.stem), []).append({"answer_id": label, "response": payload, "source_path": str(file)})
    elif path.suffix == ".jsonl":
        for row in read_jsonl(path):
            has_external_trace = False
            if traces is not None:
                key = _submission_key(row)
                if "trace" in row or any(isinstance(row.get(field), dict) and "trace" in row[field]
                                         for field in ("answer", "response", "payload")):
                    raise ValueError("Supply traces in --traces or inline in --answers, not both")
                if key in traces:
                    row = {**row, "trace": traces[key]}
                    used_traces.add(key)
                    has_external_trace = True
            try:
                response = _response_from_row(row)
            except ValueError as exc:
                raise ValueError(f"invalid answer row for {row.get('query_id')}: {exc}") from exc
            answer_id = str(row.get("answer_id") or row.get("model") or label)
            answers.setdefault(row["query_id"], []).append({
                "answer_id": answer_id, "response": response, "source_path": str(path),
                **({"trace_source_path": str(traces_path)} if has_external_trace else {}),
            })
    else:
        raise ValueError("unsupported answers path: %s" % path)
    for query_id, rows in answers.items():
        ids = [r["answer_id"] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate answer_id for %s in %s" % (query_id, label))
    if not answers:
        raise ValueError("answers source %s is empty" % spec)
    if traces is not None and set(traces) != used_traces:
        raise ValueError(f"Traces have no matching answer: {sorted(set(traces) - used_traces)}")
    return label, answers


def response_for_judge(response: Mapping[str, Any]) -> dict[str, Any]:
    return {"answer": render_answer(response),
            **{key: response[key] for key in ("sources", "limitations") if key in response}}
