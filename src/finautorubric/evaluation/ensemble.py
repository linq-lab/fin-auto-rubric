"""Majority voting over complete grading runs."""
from pathlib import Path
from typing import Any
import shutil
from .common.io import atomic_write_json, file_sha256, read_json, utc_now

ENSEMBLE_JUDGE = "ensemble_majority"
SCHEMA_VERSION = "baseline.grading-ensemble/v1"
SHARED_FROZEN = ("blind_key.jsonl", "queries.jsonl", "prompts/evaluator.md")
COPIED_FROZEN = ("blind_key.jsonl", "queries.jsonl", "prompts/evaluator.md", "rubrics", "answers")
def _display(path: Path) -> str:
    return str(path)

def _complete(run: Path) -> bool:
    summary = run / "summary.json"
    return summary.is_file() and read_json(summary).get("remaining") == 0

def merge(member_runs: list[Path], output_dir: Path) -> dict[str, Any]:
    """Write the majority-vote grading run of complete, identically blinded member runs."""

    if len(member_runs) < 3 or len(member_runs) % 2 == 0:
        raise SystemExit(f"majority voting needs an odd number of judges, at least 3 (got {len(member_runs)})")
    manifests = [read_json(run / "manifest.json") for run in member_runs]
    judges = [manifest["judge"] for manifest in manifests]
    if len(set(judges)) != len(judges):
        raise SystemExit(f"each member must use a different judge: {judges}")
    if any(manifest.get("answer_rendering") != manifests[0].get("answer_rendering") for manifest in manifests[1:]):
        raise ValueError("members differ in answer/trace rendering policy")
    incomplete = [_display(run) for run in member_runs if not _complete(run)]
    if incomplete:
        raise SystemExit(f"incomplete member runs (resume them first): {incomplete}")
    for name in SHARED_FROZEN:
        if len({file_sha256(run / "frozen" / name) for run in member_runs}) != 1:
            raise SystemExit(f"members differ in frozen/{name}: use the same --seed, --queries and --rubrics")
    jobs = manifests[0]["jobs"]

    def job_key(job: dict[str, Any]) -> tuple[str, str, str]:
        return job["answers"], job["answer_id"], job["query_id"]

    if any(sorted(map(job_key, manifest["jobs"])) != sorted(map(job_key, jobs)) for manifest in manifests[1:]):
        raise SystemExit("members differ in their answer jobs")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("ensemble output directory must be empty")
    answer_hashes = [{str(p.relative_to(run / "frozen/answers")): file_sha256(p)
                      for p in (run / "frozen/answers").rglob("*.json")} for run in member_runs]
    if any(hashes != answer_hashes[0] for hashes in answer_hashes[1:]):
        raise ValueError("members differ in frozen answers")
    first = member_runs[0]
    for rel in COPIED_FROZEN:
        source, target = first / "frozen" / rel, output_dir / "frozen" / rel
        if source.is_dir():
            shutil.copytree(source, target)
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    needed = len(judges) // 2 + 1
    unanimous = total = 0
    for job in jobs:
        rel = Path(job["answers"]) / job["answer_id"] / f"{job['query_id']}.json"
        grades = [read_json(run / "grades" / judge / rel) for run, judge in zip(member_runs, judges)]
        order = [row["criterion_id"] for row in grades[0]["verdicts"]]
        if any([row["criterion_id"] for row in grade["verdicts"]] != order for grade in grades[1:]):
            raise SystemExit(f"criterion ids differ between members for {rel}")
        verdicts = []
        for index, criterion_id in enumerate(order):
            votes = {judge: bool(grade["verdicts"][index]["pass"]) for judge, grade in zip(judges, grades)}
            passes = sum(votes.values())
            unanimous += passes in (0, len(judges))
            total += 1
            verdicts.append({"criterion_id": criterion_id, "pass": passes >= needed, "votes": votes,
                             "rationale": f"{passes}/{len(judges)} judges pass"})
        target = output_dir / "grades" / ENSEMBLE_JUDGE / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(target, {"answers": job["answers"], "answer_id": job["answer_id"], "query_id": job["query_id"],
                                   "judge": ENSEMBLE_JUDGE, "verdicts": verdicts, "graded_at": utc_now()})

    atomic_write_json(output_dir / "manifest.json", {
        "schema_version": SCHEMA_VERSION, "created_at": utc_now(), "judge": ENSEMBLE_JUDGE,
        "rule": f"a criterion passes when at least {needed} of {len(judges)} judges pass it",
        "members": [{"run": _display(run), "judge": judge, "judge_spec": manifest.get("judge_spec"),
                     "manifest_sha256": file_sha256(run / "manifest.json")}
                    for run, judge, manifest in zip(member_runs, judges, manifests)],
        "jobs": jobs, "seed": manifests[0].get("seed"), "grouping": manifests[0].get("grouping"),
        "answer_rendering": manifests[0].get("answer_rendering"),
        "frozen_hashes": {name: file_sha256(output_dir / "frozen" / name) for name in SHARED_FROZEN},
    })
    summary = {"finished_at": utc_now(), "judge": ENSEMBLE_JUDGE, "members": judges, "jobs": len(jobs),
               "graded": len(jobs), "remaining": 0, "criterion_verdicts": total,
               "unanimous_share": round(unanimous / total, 4) if total else None}
    atomic_write_json(output_dir / "summary.json", summary)
    return summary
