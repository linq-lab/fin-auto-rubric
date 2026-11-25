"""Score saved boolean verdicts with the released criterion weights."""

from collections import defaultdict
from pathlib import Path
from .common.io import atomic_write_json, read_json, read_jsonl
from .common.judge import validate_verdicts
from .common.rubric import score


def score_run(run: Path) -> dict:
    manifest = read_json(run / "manifest.json")
    if manifest.get("schema_version") == "finautorubric.judge-panel/v1":
        if (run / "ensemble/manifest.json").is_file():
            result = score_run(run / "ensemble")
        else:
            result = {"complete": False, "scale": "0-100", "aggregate": [], "per_query": [],
                      "missing_members": [judge["name"] for judge in manifest["judges"]
                                          if not (run / "members" / judge["name"] / "summary.json").is_file()
                                          or read_json(run / "members" / judge["name"] / "summary.json").get("remaining", 1) != 0]}
        atomic_write_json(run / "scores.json", result)
        return result
    keys = {r["query_id"]: r["criteria"] for r in read_jsonl(run / "frozen/blind_key.jsonl")}
    expected_queries = {r["query_id"] for r in read_jsonl(run / "frozen/queries.jsonl")}
    per_query, missing = [], []
    for job in manifest["jobs"]:
        path = run / "grades" / manifest["judge"] / job["answers"] / job["answer_id"] / f"{job['query_id']}.json"
        if not path.is_file():
            missing.append(job)
            continue
        key = keys[job["query_id"]]
        grade = read_json(path)
        validate_verdicts(grade, [r["blinded_criterion_id"] for r in key])
        verdicts = {r["criterion_id"]: r["pass"] for r in grade["verdicts"]}
        for source in sorted({r["rubric_source"] for r in key}):
            rows = [r for r in key if r["rubric_source"] == source]
            criteria = [{**r, "criterion_id": r["blinded_criterion_id"]} for r in rows]
            per_query.append({**job, "rubric_source": source, "score": 100 * score(criteria, verdicts, weighting=rows[0]["weighting"])})
    groups = defaultdict(list)
    for job in manifest["jobs"]:
        for source in {r["rubric_source"] for r in keys[job["query_id"]]}:
            groups[(job["answers"], job["answer_id"], source)]
    for row in per_query:
        groups[(row["answers"], row["answer_id"], row["rubric_source"])].append(row)
    aggregate = []
    for (answers, answer_id, source), rows in sorted(groups.items()):
        complete = {r["query_id"] for r in rows} == expected_queries
        aggregate.append({"answers": answers, "answer_id": answer_id, "rubric_source": source,
                          "questions_scored": len(rows), "questions_expected": len(expected_queries),
                          "complete": complete,
                          "mean_score": sum(r["score"] for r in rows) / len(rows) if complete else None})
    result = {"complete": not missing and bool(aggregate) and all(r["complete"] for r in aggregate),
              "scale": "0-100", "aggregation": "unweighted mean of question scores",
              "aggregate": aggregate, "per_query": per_query, "missing_jobs": missing}
    atomic_write_json(run / "scores.json", result)
    return result
