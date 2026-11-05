"""Common rubric record for every rubric source (baselines, ours, expert).

`baseline.rubric/v1`::

    {
      "schema_version": "baseline.rubric/v1",
      "method": "tick",                       # registry name, "expert", or "finautorubric"
      "weighting": "uniform|points|qworld_alpha",
      "query_id": "...", "benchmark": "...", "information_cutoff": "YYYY-MM-DD" | null,
      "criteria": [{"criterion_id": "C001", "rubric_text": "...", "weight": 1.0,
                    "polarity": "positive|negative", "meta": {...}}],
      "generation": {...author family / model / usage...},
      "intermediate": {...method-specific trace...},
      "prompt_hashes": {...}
    }

Scoring follows each source's native aggregation (§5.2 of the paper):
uniform pass share (TICK, RocketEval, EvalAgent), point-weighted pass share
(ours, Direct-Generate, expert points) and Qworld's signed-weight normalisation
(HealthBench-style: sum of satisfied weights over the positive-weight total).
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "baseline.rubric/v1"
WEIGHTINGS = ("uniform", "points", "qworld_alpha")
POLARITIES = ("positive", "negative")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def criterion_rows(
    texts_or_rows: Iterable[Any],
    *,
    default_weight: float = 1.0,
    prefix: str = "C",
) -> list[dict[str, Any]]:
    """Assign sequential ids to plain strings or partial rows."""

    rows: list[dict[str, Any]] = []
    for item in texts_or_rows:
        if isinstance(item, str):
            row: dict[str, Any] = {"rubric_text": item}
        elif isinstance(item, Mapping):
            row = dict(item)
        else:
            raise TypeError("criterion must be a string or a mapping")
        text = normalize_text(row.get("rubric_text", ""))
        if not text:
            continue
        weight = float(row.get("weight", default_weight))
        polarity = row.get("polarity") or ("negative" if weight < 0 else "positive")
        rows.append({
            "criterion_id": "%s%03d" % (prefix, len(rows) + 1),
            "rubric_text": text,
            "weight": weight,
            "polarity": polarity,
            "meta": dict(row.get("meta") or {}),
        })
    return rows


def make_record(
    *,
    method: str,
    weighting: str,
    query: Mapping[str, Any],
    criteria: Sequence[Mapping[str, Any]],
    generation: Mapping[str, Any],
    intermediate: Mapping[str, Any] | None = None,
    prompt_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    record = {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "weighting": weighting,
        "query_id": query["query_id"],
        "benchmark": query.get("benchmark"),
        "information_cutoff": query.get("information_cutoff"),
        "criteria": [dict(row) for row in criteria],
        "generation": dict(generation),
        "intermediate": dict(intermediate or {}),
        "prompt_hashes": dict(prompt_hashes or {}),
    }
    validate_record(record)
    return record


def validate_record(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected rubric schema_version: %r" % record.get("schema_version"))
    if record.get("weighting") not in WEIGHTINGS:
        raise ValueError("unknown weighting: %r" % record.get("weighting"))
    if not isinstance(record.get("query_id"), str) or not record["query_id"]:
        raise ValueError("record requires a query_id")
    criteria = record.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("record requires a non-empty criteria list")
    seen: set[str] = set()
    for row in criteria:
        for key in ("criterion_id", "rubric_text", "weight", "polarity"):
            if key not in row:
                raise ValueError("criterion missing %s" % key)
        if row["criterion_id"] in seen:
            raise ValueError("duplicate criterion_id %s" % row["criterion_id"])
        seen.add(row["criterion_id"])
        if not isinstance(row["rubric_text"], str) or not row["rubric_text"].strip():
            raise ValueError("empty rubric_text in %s" % row["criterion_id"])
        if row["polarity"] not in POLARITIES:
            raise ValueError("bad polarity in %s" % row["criterion_id"])
        weight = float(row["weight"])
        if record["weighting"] != "qworld_alpha" and weight <= 0:
            raise ValueError("non-positive weight in %s under %s weighting" % (row["criterion_id"], record["weighting"]))
        if record["weighting"] == "qworld_alpha" and weight == 0:
            raise ValueError("zero weight in %s under qworld_alpha weighting" % row["criterion_id"])


def score(criteria: Sequence[Mapping[str, Any]], verdicts: Mapping[str, bool], *, weighting: str) -> float:
    """Point-weighted pass share in [0, 1] under the requested weighting.

    `verdicts` maps criterion_id to the judge's pass decision. For a negative
    Qworld criterion, `pass` means the response exhibits the bad behaviour, so
    its (negative) weight is added to the numerator as in Qworld Eq. 3-4.
    """

    if weighting not in WEIGHTINGS:
        raise ValueError("unknown weighting: %s" % weighting)
    if not criteria:
        raise ValueError("cannot score an empty rubric")
    missing = [row["criterion_id"] for row in criteria if row["criterion_id"] not in verdicts]
    if missing:
        raise ValueError("verdicts missing for %s" % missing)
    if weighting == "uniform":
        return sum(1.0 for row in criteria if verdicts[row["criterion_id"]]) / len(criteria)
    if weighting == "points":
        total = sum(float(row["weight"]) for row in criteria)
        earned = sum(float(row["weight"]) for row in criteria if verdicts[row["criterion_id"]])
        return earned / total if total else 0.0
    positive_total = sum(float(row["weight"]) for row in criteria if float(row["weight"]) > 0)
    if positive_total <= 0:
        raise ValueError("qworld_alpha rubric has no positive-weight criterion")
    earned = sum(float(row["weight"]) for row in criteria if verdicts[row["criterion_id"]])
    return min(1.0, max(0.0, earned / positive_total))


def judge_view(criteria: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """The only fields a judge sees: an opaque id and the criterion text."""

    return [{"criterion_id": row["criterion_id"], "rubric_text": row["rubric_text"]} for row in criteria]
