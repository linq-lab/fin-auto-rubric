"""Validate public inputs and run the research criterion grader portably."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import json
import math
from pathlib import Path
import re
import tomllib
import tempfile

from . import grade
from .common.io import atomic_write_json, canonical_sha256, file_sha256, read_json, read_jsonl
from .common.judge import JUDGE_SPECS, load_answers
from .common.trace import RENDERING_POLICY
from .prompts.judge import load_judge_prompt
from .scoring import score_run
from .ensemble import merge


def safe_id(value, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"{name} must contain only letters, digits, underscores, dots or hyphens and start with a letter or digit")
    return value


DEFAULT_CONFIG = Path(__file__).with_name("judges.toml")
PANEL_SCHEMA = "finautorubric.judge-panel/v1"


def validate_judge(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Each judge must be a TOML table")
    permitted = {"name", "model", "reasoning_effort", "provider", "request_timeout_seconds", "max_tokens", "max_retries"}
    if set(raw) - permitted:
        raise ValueError(f"Unknown judge configuration keys: {sorted(set(raw) - permitted)}")
    safe_id(raw.get("name"), "judge name")
    if not isinstance(raw.get("model"), str) or not raw["model"].strip():
        raise ValueError("Judge configuration requires a model")
    if raw.get("reasoning_effort", "medium") not in {"none", "minimal", "low", "medium", "high", "xhigh"}:
        raise ValueError("Unsupported reasoning_effort")
    provider = raw.get("provider", {"require_parameters": True})
    if not isinstance(provider, dict):
        raise ValueError("provider must be a TOML table")
    for key in ("max_tokens", "max_retries"):
        if key in raw and (type(raw[key]) is not int or raw[key] <= 0):
            raise ValueError(f"{key} must be a positive integer")
    timeout = raw.get("request_timeout_seconds", 300.0)
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("request_timeout_seconds must be positive and finite")
    return {"backend": "openrouter", "reasoning_effort": "medium", "provider": provider, **raw}


def load_config(path: Path | None = None) -> list[dict]:
    raw = tomllib.loads((path or DEFAULT_CONFIG).read_text())
    if "judges" in raw:
        if set(raw) != {"judges"} or not isinstance(raw["judges"], list):
            raise ValueError("Panel configuration must contain only [[judges]] tables")
        configs = [validate_judge(judge) for judge in raw["judges"]]
    else:
        # Retain the original single-judge configuration format.
        configs = [validate_judge(raw)]
    if not configs or (len(configs) != 1 and len(configs) % 2 == 0):
        raise ValueError("Use one judge or an odd number of at least three judges")
    if len({c["name"] for c in configs}) != len(configs):
        raise ValueError("Judge names must be unique")
    if len({c["model"] for c in configs}) != len(configs):
        raise ValueError("Each judge must use a distinct model")
    return configs


def validate_inputs(queries_path: Path, rubrics_path: Path, answers_path: Path, traces_path: Path | None = None) -> dict:
    queries = read_jsonl(queries_path)
    if not queries:
        raise ValueError("Queries cannot be empty")
    by_id = {}
    for row in queries:
        qid = safe_id(row.get("query_id"), "query_id")
        if qid in by_id:
            raise ValueError(f"Duplicate query_id: {qid}")
        query_text = row.get("query", row.get("raw_query"))
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError(f"Missing query text: {qid}")
        if "query" in row and "raw_query" in row and row["query"] != row["raw_query"]:
            raise ValueError(f"Conflicting query text: {qid}")
        cutoff = row.get("information_cutoff")
        if not isinstance(cutoff, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", cutoff):
            raise ValueError(f"information_cutoff must use YYYY-MM-DD: {qid}")
        date.fromisoformat(cutoff)
        if row.get("query_date") and row["query_date"] != cutoff:
            raise ValueError(f"Conflicting query dates: {qid}")
        by_id[qid] = {**row, "query": query_text}
    rubric_ids, criteria_count = set(), 0
    for row in read_jsonl(rubrics_path):
        qid = safe_id(row.get("query_id"), "rubric query_id")
        if qid in rubric_ids or qid not in by_id:
            raise ValueError(f"Duplicate or unknown rubric query_id: {qid}")
        rubric_ids.add(qid)
        if row.get("query") is not None and row["query"] != by_id[qid]["query"]:
            raise ValueError(f"Rubric query text differs: {qid}")
        if row.get("information_cutoff") is not None and row["information_cutoff"] != by_id[qid]["information_cutoff"]:
            raise ValueError(f"Rubric query date differs: {qid}")
        criteria = row.get("rubric")
        if not isinstance(criteria, list) or not criteria:
            raise ValueError(f"Expected a nonempty rubric list: {qid}")
        ids, points = set(), []
        for criterion in criteria:
            cid = criterion.get("criterion_id")
            if not isinstance(cid, str) or not cid.strip() or cid in ids:
                raise ValueError(f"Invalid or duplicate criterion_id: {qid}")
            ids.add(cid)
            weight = criterion.get("points")
            if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"Criterion points must be positive and finite: {qid}/{cid}")
            if not isinstance(criterion.get("rubric_text"), str) or not criterion["rubric_text"].strip():
                raise ValueError(f"Missing rubric_text: {qid}/{cid}")
            points.append(weight)
        if not math.isclose(math.fsum(points), 100, abs_tol=1e-7, rel_tol=0):
            raise ValueError(f"Rubric points must total 100: {qid}")
        criteria_count += len(criteria)
    if rubric_ids != set(by_id):
        raise ValueError("Every query must have exactly one rubric")
    # This public release grades the embedded criteria; it never silently ignores
    # an internal scoring-policy sidecar with different evaluation semantics.
    if (rubrics_path.parent / "scoring_policy.json").exists():
        raise ValueError("Use the public JSONL export; internal scoring_policy.json bundles are not supported")
    _, answers = load_answers(f"answers={answers_path}", traces_path=traces_path)
    if set(answers) != set(by_id):
        raise ValueError("Answer query IDs must match the input queries exactly")
    answer_ids = None
    for qid, entries in answers.items():
        current = {safe_id(entry["answer_id"], "answer_id") for entry in entries}
        if answer_ids is not None and current != answer_ids:
            raise ValueError("Every answer_id must have an answer for every query")
        answer_ids = current
    return {"questions": len(by_id), "criteria": criteria_count, "answer_sets": len(answer_ids or []),
            "answer_count": sum(len(v) for v in answers.values())}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fin-auto-rubric evaluate", description="Grade public rubrics against supplied answers and agent traces.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--dataset", type=Path, help="Combined public question/rubric JSONL")
    inputs.add_argument("--queries", type=Path, help="Questions JSONL; requires --rubrics")
    parser.add_argument("--rubrics", type=Path, help="Rubrics JSONL with criterion IDs, text, and points")
    parser.add_argument("--answers", type=Path, required=True, help="Final answers JSONL, keyed by query_id and answer_id")
    parser.add_argument("--traces", type=Path, help="Optional agent traces JSONL, matched by query_id/answer_id; missing traces may lower scores when required evidence is absent")
    parser.add_argument("--judge-config", type=Path,
                        help="TOML judge configuration; defaults to the packaged three-judge panel")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, help="Concurrent answer jobs per judge (default: 4)")
    parser.add_argument("--timeout-seconds", type=float, default=1200)
    parser.add_argument("--seed", type=int, default=77512)
    parser.add_argument("--dry-run", action="store_true", help="Validate and freeze inputs without API calls")
    parser.add_argument("--resume", action="store_true", help="Explicitly resume an existing run with identical inputs/configuration")
    return parser


def _fingerprint(queries, rubrics, answers, seed, traces=None) -> dict:
    return {"queries": file_sha256(queries), "rubrics": file_sha256(rubrics),
            "answers": file_sha256(answers), "seed": seed,
            "prompt": canonical_sha256(load_judge_prompt()), "batch_size": grade.MAX_CRITERIA_PER_CALL,
            "answer_rendering": RENDERING_POLICY,
            **({"traces": file_sha256(traces)} if traces is not None else {})}


def _check_output(out: Path, fingerprint: dict, resume: bool) -> dict | None:
    if out.exists() and any(out.iterdir()):
        if not resume:
            raise ValueError("Output already exists; use --resume with identical inputs or choose a new directory")
        request = out / "request.json"
        if not request.is_file() or read_json(request) != fingerprint:
            raise ValueError("Inputs or judge configuration changed; choose a new output directory")
        if not (out / "manifest.json").is_file():
            raise ValueError("Incomplete preparation; choose a new output directory")
        return read_json(out / "manifest.json")
    if resume:
        raise ValueError("Nothing to resume at the supplied output directory")
    return None


def _prepare_member(args, queries, rubrics, answers, config, out, fingerprint):
    request = {**fingerprint, "judge": config}
    previous = _check_output(out, request, args.resume)
    if previous is not None:
        if (previous.get("judge") != config["name"] or previous.get("judge_spec") != config
                or previous.get("seed") != args.seed):
            raise ValueError("Frozen judge configuration changed; choose a new output directory")
        if not previous.get("public_frozen_hashes"):
            raise ValueError("Run is missing its frozen-input checks")
        for relative, digest in previous["public_frozen_hashes"].items():
            path = (out / relative).resolve()
            if not path.is_relative_to(out) or not path.is_file() or file_sha256(path) != digest:
                raise ValueError(f"Frozen input changed: {relative}")
    JUDGE_SPECS[config["name"]] = config
    internal = argparse.Namespace(queries=queries, rubrics=[f"finautorubric={rubrics}"], answers=[f"answers={answers}"],
                                  traces=args.traces,
                                  judge=config["name"], output_dir=out, workers=args.workers,
                                  timeout_seconds=args.timeout_seconds, seed=args.seed, max_jobs=None,
                                  grouping="per_source", require_all_sources=True, verdict_schema="pass_rationale/v2",
                                  evaluator_prompt=None, replay_dir=None)
    manifest = grade.prepare(internal)
    if previous is None:
        manifest["public_frozen_hashes"] = {str(p.relative_to(out)): file_sha256(p)
                                            for p in (out / "frozen").rglob("*") if p.is_file()}
        atomic_write_json(out / "manifest.json", manifest)
        atomic_write_json(out / "request.json", request)
    grade.validate_per_source_run(out, manifest)
    return internal, manifest


def _tree_hashes(root: Path) -> dict:
    return {str(p.relative_to(root)): file_sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def _check_ensemble(out: Path, panel: dict, members: list[Path]) -> None:
    if not (out / "ensemble").exists():
        if panel.get("ensemble_output_hashes"):
            raise ValueError("Saved ensemble output is missing")
        return
    member_hashes = {member.name: _tree_hashes(member / "grades") for member in members}
    if (panel.get("member_grade_hashes") != member_hashes
            or panel.get("ensemble_output_hashes") != _tree_hashes(out / "ensemble")):
        raise ValueError("Saved ensemble or member grades changed; use a fresh output directory")


async def _run_panel(prepared) -> list:
    # All judges share one event loop and an HTTP thread pool sized for the panel.
    workers = sum(args.workers for args, _ in prepared)
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=workers))
    return await asyncio.gather(*(grade.grade(args, manifest, manage_executor=False)
                                  for args, manifest in prepared), return_exceptions=True)


def evaluate(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.queries and not args.rubrics:
        raise ValueError("--queries requires --rubrics")
    if args.dataset and args.rubrics:
        raise ValueError("--dataset already includes rubrics; omit --rubrics")
    if args.workers < 1 or not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise ValueError("workers and timeout-seconds must be positive")
    queries = (args.dataset or args.queries).expanduser().resolve()
    rubrics = (args.rubrics or args.dataset).expanduser().resolve()
    answers = args.answers.expanduser().resolve()
    if not answers.is_file() or answers.suffix != ".jsonl":
        raise ValueError("--answers must be a JSONL file")
    args.traces = args.traces.expanduser().resolve() if args.traces is not None else None
    if args.traces is not None and (not args.traces.is_file() or args.traces.suffix != ".jsonl"):
        raise ValueError("--traces must be a JSONL file")
    configs = load_config(args.judge_config.expanduser() if args.judge_config else None)
    if not args.dry_run and any("YOUR_" in config["model"] for config in configs):
        raise ValueError("Replace the placeholder model in --judge-config before a live run")
    counts = validate_inputs(queries, rubrics, answers, args.traces)
    out = args.output_dir.expanduser().resolve()
    fingerprint = _fingerprint(queries, rubrics, answers, args.seed, args.traces)

    if len(configs) == 1:
        internal, manifest = _prepare_member(args, queries, rubrics, answers, configs[0], out, fingerprint)
        if args.dry_run:
            print(json.dumps({"status": "prepared", "model_calls": 0, **counts,
                              "judges": [configs[0]["name"]], "output_dir": str(out)}, indent=2))
            return 0
        result = asyncio.run(grade.grade(internal, manifest))
        scores = score_run(out)
        print(json.dumps({"complete": scores["complete"], "scores": str(out / "scores.json")}, indent=2))
        return result

    request = {**fingerprint, "judges": configs, "aggregation": "criterion_majority"}
    previous = _check_output(out, request, args.resume)
    if previous is not None and (previous.get("schema_version") != PANEL_SCHEMA or previous.get("judges") != configs):
        raise ValueError("Frozen judge panel changed; choose a new output directory")
    # Prepare and validate every member before any model call.
    prepared = [_prepare_member(args, queries, rubrics, answers, config, out / "members" / config["name"], fingerprint)
                for config in configs]
    members = [internal.output_dir for internal, _ in prepared]
    panel = previous or {"schema_version": PANEL_SCHEMA, "judges": configs, "seed": args.seed,
                         "aggregation": "criterion_majority", "members": [str(p.relative_to(out)) for p in members]}
    if previous is None:
        atomic_write_json(out / "request.json", request)
        atomic_write_json(out / "manifest.json", panel)
    _check_ensemble(out, panel, members)
    if args.dry_run:
        print(json.dumps({"status": "prepared", "model_calls": 0, **counts,
                          "judges": [c["name"] for c in configs], "output_dir": str(out)}, indent=2))
        return 0

    outcomes = asyncio.run(_run_panel(prepared))
    statuses = []
    for member, (internal, manifest), outcome in zip(members, prepared, outcomes):
        scores = score_run(member)
        status = {"judge": internal.judge, "complete": outcome == 0 and scores["complete"],
                  "jobs": len(manifest["jobs"]), "remaining": len(scores["missing_jobs"])}
        if isinstance(outcome, BaseException):
            status["error"] = str(outcome)
        statuses.append(status)
    summary = {"complete": False, "judges": len(configs), "completed_judges": sum(s["complete"] for s in statuses),
               "members": statuses, "aggregation": "criterion_majority"}
    atomic_write_json(out / "summary.json", summary)
    if not all(s["complete"] for s in statuses):
        atomic_write_json(out / "scores.json", {"complete": False, "scale": "0-100", "aggregate": [], "per_query": [],
                                                "missing_members": [s["judge"] for s in statuses if not s["complete"]]})
        print(json.dumps(summary, indent=2))
        return 2

    if not (out / "ensemble").exists():
        # Publish the derived ensemble only after the whole merge succeeds.
        with tempfile.TemporaryDirectory(prefix=".ensemble-", dir=out) as staging:
            destination = Path(staging) / "result"
            merge(members, destination)
            score_run(destination)
            destination.rename(out / "ensemble")
        panel["member_grade_hashes"] = {member.name: _tree_hashes(member / "grades") for member in members}
        panel["ensemble_output_hashes"] = _tree_hashes(out / "ensemble")
        atomic_write_json(out / "manifest.json", panel)
    scores = score_run(out)
    summary["complete"] = scores["complete"]
    atomic_write_json(out / "summary.json", summary)
    print(json.dumps({"complete": scores["complete"], "judges": len(configs), "scores": str(out / "scores.json")}, indent=2))
    return 0 if scores["complete"] else 2
