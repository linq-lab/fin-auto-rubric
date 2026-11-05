#!/usr/bin/env python3
"""Criterion-level grading with frozen inputs, blinded batches, and resumable results."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from finautorubric.evaluation.common import schemas as S  # noqa: E402
from finautorubric.evaluation.common.backends import BackendError, ReplayBackend  # noqa: E402
from finautorubric.evaluation.common.io import atomic_write_json, canonical_sha256, file_sha256, read_json, read_jsonl, utc_now, write_jsonl  # noqa: E402
from finautorubric.evaluation.common.judge import DEFAULT_JUDGE, JUDGE_SPECS, evaluator_schema, load_answers, load_rubric_source, make_judge, response_for_judge, validate_verdicts  # noqa: E402
from finautorubric.evaluation.common.trace import RENDERING_POLICY
from finautorubric.evaluation.prompts.judge import load_judge_prompt, load_legacy_judge_prompt  # noqa: E402

MAX_CRITERIA_PER_CALL = 30
GROUPING = "per_source"
CAPACITY_ERROR = re.compile(r"\b(?:at capacity|capacity|overloaded|temporarily unavailable|rate limit(?:ed)?)\b", re.IGNORECASE)


async def generate_with_capacity_retry(backend, *, stage: str, **kwargs):
    """Retry one transient capacity failure without reusing its trace path."""

    try:
        return await backend.generate(stage=stage, **kwargs)
    except BackendError as exc:
        if not CAPACITY_ERROR.search(str(exc)):
            raise
        await asyncio.sleep(1.0)
        return await backend.generate(stage=stage + "_capacity_retry", **kwargs)


VALIDATION_ATTEMPTS = 3


def restore_batch_order(output: dict[str, Any], ids: list[str]) -> bool:
    """Reorder a complete, duplicate-free verdict list into batch order; True if it moved anything.

    Criterion ids are opaque and unique within a batch, so a verdict set that covers exactly
    the batch ids carries the same judgments in any order. Anything else is left for
    `validate_verdicts` to reject.
    """

    verdicts = output.get("verdicts")
    if not isinstance(verdicts, list):
        return False
    got = [v.get("criterion_id") if isinstance(v, dict) else None for v in verdicts]
    if got == ids or len(got) != len(ids) or set(got) != set(ids) or len(set(got)) != len(got):
        return False
    by_id = {v["criterion_id"]: v for v in verdicts}
    output["verdicts"] = [by_id[cid] for cid in ids]
    return True

# Verdict schema (2026-09-08): the `evidence` array is dropped. Across the bfb50/fab27 grading runs it
# was ~1 item per criterion with `source_url == "criterion_reference"` in 98-99% of items, i.e. a
# restatement of `rationale`, and `compute_table2.py` reads only `pass`. Defined here rather than in
# `common/judge.py`, which is hashed into the `run_baselines.py` config lock.
# A run's manifest records its `verdict_schema`; manifests without the key are `evidence/v1`, so the
# frozen prompt (which for v1 still asks for evidence) always matches the schema sent to the judge.
# Since 2026-09-16 new v2 runs freeze `prompts/evaluator.md`; NO_EVIDENCE_OVERLAY is only appended when
# `migrate_grading_verdicts.py` moves a v1 run, whose frozen v6.5 prompt still asks for evidence, to v2.
# The judge payload's shape is covered by no frozen hash, so the manifest records it. `context/v1` is
# the pre-2026-09-18 payload (see `evaluator_context`); manifests without the key are `context/v1`.
PAYLOAD_SCHEMA = "context/v2"
LEGACY_VERDICT_SCHEMA = "evidence/v1"
DEFAULT_VERDICT_SCHEMA = "pass_rationale/v2"
VERDICT_SCHEMAS = (LEGACY_VERDICT_SCHEMA, DEFAULT_VERDICT_SCHEMA)
NO_EVIDENCE_OVERLAY = """
# Verdict format

Each verdict carries only `criterion_id`, `pass`, and `rationale`; there is no
`evidence` array and base rule 7 does not apply. State the decisive check in
`rationale` in one or two sentences: the response's value or claim set against
the criterion's reference content.
""".strip()


def verdict_schema(ids: list[str], version: str = DEFAULT_VERDICT_SCHEMA) -> dict[str, Any]:
    if version == LEGACY_VERDICT_SCHEMA:
        return evaluator_schema(ids)
    if version == DEFAULT_VERDICT_SCHEMA:
        verdict = S.obj({"criterion_id": S.string(enum=ids), "pass": S.BOOLEAN, "rationale": S.string(min_length=1)})
        return S.obj({"verdicts": S.arr(verdict, min_items=len(ids), max_items=len(ids))})
    raise ValueError(f"unknown verdict schema: {version}")


def judge_prompt(version: str = DEFAULT_VERDICT_SCHEMA) -> str:
    """Judge prompt frozen into a new run: `prompts/evaluator.md` for v2, the legacy v6.5 evaluator for v1."""

    if version == LEGACY_VERDICT_SCHEMA:
        return load_legacy_judge_prompt()
    if version == DEFAULT_VERDICT_SCHEMA:
        return load_judge_prompt()
    raise ValueError(f"unknown verdict schema: {version}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--rubrics", action="append", required=True, metavar="LABEL=PATH[:METHOD]")
    parser.add_argument("--answers", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--judge", choices=sorted(JUDGE_SPECS), default=DEFAULT_JUDGE,
                        help=f"fixed OpenRouter judge (default: {DEFAULT_JUDGE})")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=77512)
    parser.add_argument("--grouping", choices=(GROUPING,), default=GROUPING,
                        help="one rubric source per judge call; per_source is the only supported layout")
    parser.add_argument("--require-all-sources", action="store_true", help="skip queries missing any rubric source (default: skip only missing sources)")
    parser.add_argument("--verdict-schema", choices=VERDICT_SCHEMAS, default=None,
                        help=f"verdict fields sent to the judge (new runs default to {DEFAULT_VERDICT_SCHEMA}; an existing run keeps the schema in its manifest)")
    parser.add_argument("--evaluator-prompt", type=Path, default=None,
                        help="optional evaluator prompt file to freeze for a new pass_rationale/v2 run")
    parser.add_argument("--replay-dir", type=Path, default=None, help="tests: grades/{judge}/{answers}/{answer_id}/{query_id}.json fixtures")
    return parser.parse_args(argv)


def blind(query_id: str, seed: int, sources: dict[str, dict[str, Any]], grouping: str = GROUPING) -> tuple[list[dict[str, str]], list[dict[str, Any]], list[list[str]]]:
    """Opaque criterion ids for one query plus the judge batches (lists of ids).

    Sources are in seeded order, criteria shuffled within each source, and batches
    cut inside a source so a call never mixes two rubrics. The batches are consecutive
    slices of the returned order, so a grade file
    (verdicts in batch order) lines up with `blind_key.jsonl`.
    """

    if grouping != GROUPING:
        raise ValueError("only per_source grading is supported")
    arms = []
    for label, record in sources.items():
        arms.append((label, record["weighting"], [
            {"original_criterion_id": row["criterion_id"], "rubric_text": row["rubric_text"], "weight": float(row["weight"]), "polarity": row["polarity"]}
            for row in record["criteria"]
        ]))
    rng = random.Random(int(hashlib.sha256(f"{seed}:{query_id}".encode()).hexdigest(), 16))
    rng.shuffle(arms)
    segments = []
    for label, weighting, rows in arms:
        rows = list(rows)
        rng.shuffle(rows)
        segments.append([(label, weighting, row) for row in rows])
    public, key, batches = [], [], []
    index = 0
    for segment in segments:
        ids = []
        for label, weighting, row in segment:
            index += 1
            blinded_id = f"C{index:03d}"
            ids.append(blinded_id)
            public.append({"criterion_id": blinded_id, "rubric_text": row["rubric_text"]})
            key.append({"blinded_criterion_id": blinded_id, "rubric_source": label, "weighting": weighting, **row})
        batches.extend(ids[start: start + MAX_CRITERIA_PER_CALL] for start in range(0, len(ids), MAX_CRITERIA_PER_CALL))
    return public, key, batches


def evaluator_context(query: dict[str, Any]) -> dict[str, Any]:
    """Query context shown to the judge: identity, the query, and its date.

    Until 2026-09-18 the payload also carried `temporal_context`,
    `finance_domain_context`, `context_provenance` (and a constant
    `candidate_slot` beside it), copied from the v6.5 contextualizer output. Only
    `temporal_context.information_cutoff` was ever read, by rule 3 of the legacy
    v6.5 evaluator; `prompts/evaluator.md` reads the cutoff off `query_date`
    instead, and the other three were referenced by no prompt. `benchmark` also
    told the judge which benchmark an item came from, which the blinding did not
    intend. Do not pair this payload with `--verdict-schema evidence/v1`: that
    prompt still names `temporal_context.information_cutoff`.
    """

    cutoff = query.get("information_cutoff") or query.get("query_date") or "unknown"
    if query.get("information_cutoff") and query.get("query_date") and query["information_cutoff"] != query["query_date"]:
        raise ValueError("conflicting query dates")
    return {"query_id": query["query_id"], "query": query["query"], "query_date": query.get("query_date") or cutoff}


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.grouping != GROUPING:
        raise SystemExit("only per_source grading is supported")
    out = args.output_dir
    manifest_path = out / "manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest.get("grouping") != GROUPING:
            raise SystemExit("historical interleaved or unspecified grouping cannot be resumed; regrade frozen inputs in a fresh per_source directory")
        if manifest["queries_sha256"] != file_sha256(args.queries.resolve()) or manifest["judge"] != args.judge:
            raise SystemExit("output directory belongs to a different grading configuration; use a fresh directory")
        if args.verdict_schema and manifest.get("verdict_schema", LEGACY_VERDICT_SCHEMA) != args.verdict_schema:
            raise SystemExit("output directory was prepared with verdict schema %s; use a fresh directory" % manifest.get("verdict_schema", LEGACY_VERDICT_SCHEMA))
        if args.evaluator_prompt and file_sha256(args.evaluator_prompt.resolve()) != manifest["frozen_hashes"]["evaluator.md"]:
            raise SystemExit("evaluator prompt differs from the frozen run; use a fresh directory")
        return manifest
    version = args.verdict_schema or DEFAULT_VERDICT_SCHEMA
    if args.evaluator_prompt and version != DEFAULT_VERDICT_SCHEMA:
        raise SystemExit("--evaluator-prompt requires pass_rationale/v2")
    queries = {row["query_id"]: {**row, "query": row.get("query", row.get("raw_query"))}
               for row in read_jsonl(args.queries)}
    sources = dict(load_rubric_source(spec) for spec in args.rubrics)
    answers = dict(load_answers(spec, traces_path=getattr(args, "traces", None)) for spec in args.answers)
    frozen = out / "frozen"
    (frozen / "prompts").mkdir(parents=True, exist_ok=True)
    prompt = args.evaluator_prompt.read_text(encoding="utf-8") if args.evaluator_prompt else judge_prompt(version)
    (frozen / "prompts/evaluator.md").write_text(prompt, encoding="utf-8")
    write_jsonl(frozen / "queries.jsonl", list(queries.values()))
    for label, records in sources.items():
        write_jsonl(frozen / "rubrics" / f"{label}.jsonl", [records[q] for q in sorted(records)])
    blind_key, jobs, skipped = [], [], []
    for query_id, query in queries.items():
        present = {label: records[query_id] for label, records in sources.items() if query_id in records}
        missing = sorted(set(sources) - set(present))
        if not present or (args.require_all_sources and missing):
            skipped.append({"query_id": query_id, "missing_sources": missing})
            continue
        public, key, batches = blind(query_id, args.seed, present, args.grouping)
        atomic_write_json(frozen / "blinded" / f"{query_id}.json", {"query_id": query_id, "criteria": public, "batches": batches, "missing_sources": missing})
        blind_key.append({"query_id": query_id, "criteria": key})
        for answers_label, per_query in answers.items():
            for answer in per_query.get(query_id, []):
                jobs.append({"answers": answers_label, "answer_id": answer["answer_id"], "query_id": query_id})
                atomic_write_json(frozen / "answers" / answers_label / answer["answer_id"] / f"{query_id}.json",
                                  {"query_id": query_id, "response": answer["response"],
                                   "candidate_response": response_for_judge(answer["response"]),
                                   "source_path": answer["source_path"],
                                   **({"trace_source_path": answer["trace_source_path"]} if "trace_source_path" in answer else {})})
    write_jsonl(frozen / "blind_key.jsonl", blind_key)
    manifest = {
        "schema_version": "baseline.grading/v1", "created_at": utc_now(), "judge": args.judge, "judge_spec": JUDGE_SPECS[args.judge],
        "queries_path": str(args.queries.resolve()), "queries_sha256": file_sha256(args.queries.resolve()),
        "rubric_sources": {label: {"spec": spec, "queries": len(records), "weighting": sorted({r["weighting"] for r in records.values()})} for spec, (label, records) in zip(args.rubrics, sources.items())},
        "answer_sources": {label: {"queries": len(per), "answers": sum(len(v) for v in per.values())} for label, per in answers.items()},
        "jobs": jobs, "skipped_queries": skipped, "seed": args.seed, "max_criteria_per_call": MAX_CRITERIA_PER_CALL, "grouping": args.grouping,
        "verdict_schema": version, "payload_schema": PAYLOAD_SCHEMA,
        "answer_rendering": RENDERING_POLICY,
        "blinding": "opaque criterion ids; one rubric source per judge call, criteria shuffled within the source; source labels withheld from the judge",
        "frozen_hashes": {"evaluator.md": file_sha256(frozen / "prompts/evaluator.md"), "blind_key.jsonl": file_sha256(frozen / "blind_key.jsonl")},
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def validate_per_source_run(out: Path, manifest: dict[str, Any]) -> dict[str, list[list[str]]]:
    """Check every frozen batch before opening a backend or reusing cached grades."""
    if manifest.get("grouping") != GROUPING:
        raise ValueError("only per_source runs may be graded; regrade historical inputs in a fresh directory")
    keys = {row["query_id"]: row["criteria"] for row in read_jsonl(out / "frozen/blind_key.jsonl")}
    plans = {}
    for query_id in {job["query_id"] for job in manifest["jobs"]}:
        doc = read_json(out / "frozen/blinded" / f"{query_id}.json")
        ids = [row["criterion_id"] for row in doc["criteria"]]
        key = keys[query_id]
        plan = doc.get("batches")
        if not plan or [cid for batch in plan for cid in batch] != ids or len(set(ids)) != len(ids):
            raise ValueError(f"invalid or missing per_source batch plan: {query_id}; use a fresh directory")
        if [row["blinded_criterion_id"] for row in key] != ids:
            raise ValueError(f"blind key order mismatch: {query_id}")
        source_of = {row["blinded_criterion_id"]: row["rubric_source"] for row in key}
        for batch in plan:
            if not batch or len(batch) > MAX_CRITERIA_PER_CALL or len({source_of[cid] for cid in batch}) != 1:
                raise ValueError(f"batch mixes rubric sources or exceeds the batch limit: {query_id}; regrade in a fresh directory")
        plans[query_id] = plan
    return plans


async def grade(args: argparse.Namespace, manifest: dict[str, Any], *, manage_executor: bool = True) -> int:
    # OpenRouter's stdlib HTTP transport runs through asyncio.to_thread. Match its
    # thread pool to --workers so values above Python's small default are real.
    if manage_executor:
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=max(1, args.workers)))
    out = args.output_dir
    batch_plans = validate_per_source_run(out, manifest)
    queries = {row["query_id"]: row for row in read_jsonl(out / "frozen/queries.jsonl")}
    prompt = (out / "frozen/prompts/evaluator.md").read_text(encoding="utf-8")
    judge_key = manifest["judge"]
    version = manifest.get("verdict_schema", LEGACY_VERDICT_SCHEMA)
    jobs = list(manifest["jobs"])
    random.Random(manifest["seed"]).shuffle(jobs)

    def grade_path(job):
        return out / "grades" / judge_key / job["answers"] / job["answer_id"] / f"{job['query_id']}.json"

    pending = [job for job in jobs if not grade_path(job).is_file()]
    if args.max_jobs is not None:
        pending = pending[: args.max_jobs]
    backend = (make_judge(judge_key, timeout_seconds=args.timeout_seconds, trace_dir=out / "traces" / judge_key)
               if pending and not args.replay_dir else None)
    print(f"judge={judge_key} verdict_schema={version} jobs={len(jobs)} pending={len(pending)}", flush=True)
    semaphore = asyncio.Semaphore(max(1, args.workers))
    done = 0

    async def one(job):
        nonlocal done
        query_id = job["query_id"]
        blinded_doc = read_json(out / "frozen/blinded" / f"{query_id}.json")
        blinded = blinded_doc["criteria"]
        ids = [row["criterion_id"] for row in blinded]
        by_id = {row["criterion_id"]: row for row in blinded}
        batch_plan = batch_plans[query_id]
        answer = read_json(out / "frozen/answers" / job["answers"] / job["answer_id"] / f"{query_id}.json")
        payload = {"evaluation_input": evaluator_context(queries[query_id]),
                   "candidate_response": answer["candidate_response"], "criteria": blinded}
        verdicts, metadata = [], []
        async with semaphore:
            for number, batch_ids in enumerate(batch_plan, start=1):
                batch = [by_id[cid] for cid in batch_ids]
                batch_path = out / "grade_batches" / judge_key / job["answers"] / job["answer_id"] / query_id / f"batch_{number:03d}.json"
                if batch_path.is_file():
                    saved = read_json(batch_path)
                    validate_verdicts(saved["output"], batch_ids)
                    verdicts.extend(saved["output"]["verdicts"]); metadata.append(saved["metadata"])
                    continue
                if args.replay_dir:
                    fixture = read_json(args.replay_dir / "grades" / judge_key / job["answers"] / job["answer_id"] / f"{query_id}.json")
                    replay = ReplayBackend({"grade": [{"verdicts": [v for v in fixture["verdicts"] if v["criterion_id"] in batch_ids]}]})
                    result = await replay.generate(stage="grade", system_prompt=prompt, prompt="", output_schema=verdict_schema(batch_ids, version), allow_web=False)
                else:
                    stage = f"grade_{job['answers']}_{job['answer_id']}_{query_id}_b{number:03d}"
                    for attempt in range(1, VALIDATION_ATTEMPTS + 1):
                        result = await generate_with_capacity_retry(
                            backend,
                            stage=stage if attempt == 1 else f"{stage}_vretry{attempt - 1}", system_prompt=prompt,
                            prompt=("Evaluate the following anonymous session input. The criterion IDs are intentionally opaque. "
                                    "Judge every criterion independently and preserve the supplied order.\n\n# Session input\n\n```json\n"
                                    + json.dumps({**payload, "criteria": batch}, ensure_ascii=False, indent=2) + "\n```\n"),
                            output_schema=verdict_schema(batch_ids, version), allow_web=False,
                        )
                        if restore_batch_order(result.output, batch_ids):
                            result.metadata["verdicts_reordered"] = True
                        try:
                            validate_verdicts(result.output, batch_ids)
                            break
                        except ValueError:
                            if attempt == VALIDATION_ATTEMPTS:
                                raise
                    result.metadata["validation_attempts"] = attempt
                validate_verdicts(result.output, batch_ids)
                atomic_write_json(batch_path, {"output": result.output, "metadata": result.metadata})
                verdicts.extend(result.output["verdicts"]); metadata.append(result.metadata)
        validate_verdicts({"verdicts": verdicts}, ids)
        atomic_write_json(grade_path(job), {**job, "judge": judge_key, "verdicts": verdicts, "batch_metadata": metadata, "graded_at": utc_now()})
        done += 1
        print(f"progress={done}/{len(pending)} {job['answers']}/{job['answer_id']}/{query_id}", flush=True)

    results = await asyncio.gather(*(one(job) for job in pending), return_exceptions=True)
    failures = [r for r in results if isinstance(r, BaseException)]
    for failure in failures:
        print(f"failure: {failure!r}", file=sys.stderr, flush=True)
    remaining = [job for job in jobs if not grade_path(job).is_file()]
    atomic_write_json(out / "summary.json", {"finished_at": utc_now(), "judge": judge_key, "grouping": GROUPING, "verdict_schema": version,
                      "jobs": len(jobs),
                      "graded": len(jobs) - len(remaining), "remaining": len(remaining), "failures_this_run": len(failures)})
    print(f"graded={len(jobs) - len(remaining)}/{len(jobs)} failures={len(failures)}", flush=True)
    return 0 if not remaining else 2


def main(argv=None) -> int:
    args = parse_args(argv)
    manifest = prepare(args)
    return asyncio.run(grade(args, manifest))


if __name__ == "__main__":
    raise SystemExit(main())
