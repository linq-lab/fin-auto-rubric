"""Download rubrics, grade supplied answers and traces, and aggregate scores."""

import argparse
import json
from pathlib import Path
import shutil
import sys

from .evaluation.common.backends import BackendError


def download(argv) -> int:
    parser = argparse.ArgumentParser(prog="fin-auto-rubric download")
    parser.add_argument("--output", type=Path, default=Path("data/finautorubric_public.jsonl"))
    parser.add_argument("--revision", help="Optional dataset commit or revision")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError(f"Output already exists: {args.output}")
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise ValueError("Install dataset support: pip install -e '.[dataset]'") from exc
    repo = "Linq-AI-Research/FinAutoRubric"
    revision = HfApi().dataset_info(repo, revision=args.revision).sha
    source = hf_hub_download(repo_id=repo, repo_type="dataset", filename="finautorubric_public.jsonl", revision=revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as target, open(source, "rb") as handle:
        shutil.copyfileobj(handle, target)
    print(json.dumps({"dataset": repo, "revision": revision, "output": str(args.output)}, indent=2))
    return 0


def merge(argv) -> int:
    from .evaluation.ensemble import merge as merge_runs
    from .evaluation.scoring import score_run
    parser = argparse.ArgumentParser(prog="fin-auto-rubric merge")
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    merge_runs(args.runs, args.output_dir)
    print(json.dumps(score_run(args.output_dir), indent=2))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="fin-auto-rubric", description=__doc__)
    parser.add_argument("command", nargs="?", choices=["download", "evaluate", "merge", "score"])
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    try:
        command, rest = argv[0], argv[1:]
        if command == "evaluate":
            from .evaluation.public import evaluate
            return evaluate(rest)
        if command in {"download", "merge"}:
            return {"download": download, "merge": merge}[command](rest)
        if command == "score":
            from .evaluation.scoring import score_run
            scorer = argparse.ArgumentParser(prog="fin-auto-rubric score")
            scorer.add_argument("--run-dir", type=Path, required=True)
            args = scorer.parse_args(rest)
            report = score_run(args.run_dir)
            print(json.dumps(report, indent=2))
            return 0 if report["complete"] else 2
        parser.error(f"Unknown command: {command}")
    except (ValueError, FileNotFoundError, BackendError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2
