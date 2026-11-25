"""The criterion judge prompt shipped with this release."""
from pathlib import Path
EVALUATOR_PATH = Path(__file__).with_name("evaluator.md")
def load_judge_prompt() -> str:
    return EVALUATOR_PATH.read_text(encoding="utf-8")
def load_legacy_judge_prompt() -> str:
    raise ValueError("Legacy grading runs are not supported; prepare a fresh run.")
