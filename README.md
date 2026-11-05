# FinAutoRubric

[![arXiv](https://img.shields.io/badge/arXiv-2609.35744-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.35744)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/Linq-AI-Research/FinAutoRubric)
[![Contact](https://img.shields.io/badge/%E2%9C%89%EF%B8%8F%20Contact-hoyounglee%40linqalpha.com-blue)](mailto:hoyounglee@linqalpha.com)

An LLM grader for the **FinAutoRubric benchmark**. Submit your agents' final answers and optional execution traces alongside the released rubrics. Three judges evaluate each criterion against the submitted evidence in parallel. Their verdicts are combined by majority vote to produce point-weighted scores.

The benchmark contains **100 questions and 4,980 weighted criteria** drawn from financial analyst workflows. Each question totals 100 points.

## What's here

| Path | Contents |
| --- | --- |
| `src/finautorubric/` | Public CLI, criterion grader, majority voting, and score aggregation |
| `src/finautorubric/evaluation/prompts/` | Criterion judge prompt |
| `examples/` | Configuration templates and synthetic input examples |
| `docs/evaluation.md` | Input validation, judge settings, and resume behavior |
| `tests/` | Offline regression and integration tests |

## Install

Requires **Python 3.11+** and an OpenRouter API key for grading. The benchmark downloader uses the optional Hugging Face dependency.

```bash
git clone https://github.com/linq-lab/fin-auto-rubric.git
cd fin-auto-rubric
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dataset]'
```

## Get the data

The [Hugging Face dataset](https://huggingface.co/datasets/Linq-AI-Research/FinAutoRubric) contains the questions and rubrics. Follow the access instructions on the dataset page, then download with an authenticated account:

```bash
hf auth login
fin-auto-rubric download --output data/finautorubric_public.jsonl
```

Use `--revision COMMIT` to select a dataset revision. Run your own agent and prepare its final answers, optionally including execution traces, in the [input format](#input-formats) below. This package grades the submitted records; it does not run your agent. Keep the rubrics separate from the answering agent and use them only for grading.

## Run the grader

### 1. Set your API key

```bash
export OPENROUTER_API_KEY="your-api-key"
```

The grader also reads this variable from a local `.env` file.

### 2. Choose your judges

The default panel uses the three models from the evaluation experiments, all with `medium` reasoning effort:

| Judge | OpenRouter model |
| --- | --- |
| Muse Spark 1.3 | `meta/muse-spark-1.3-contributor` |
| GLM-5.3-Flash | `z-ai/glm-5.3-flash` |
| DeepSeek V4.1-Flash | `deepseek/deepseek-v4.1-flash` |

No configuration flag is needed for this panel. To change models, edit the `[[judges]]` entries in [examples/judge.toml](examples/judge.toml) and pass `--judge-config examples/judge.toml`. Models must support strict structured outputs and the selected reasoning effort. For a single judge, use [examples/judge-single.toml](examples/judge-single.toml).

### 3. Grade your answers and traces

Prepare `answers.jsonl` for final answers. To include execution traces, also prepare `traces.jsonl` and pass it with the optional `--traces` argument:

```bash
fin-auto-rubric evaluate \
  --dataset data/finautorubric_public.jsonl \
  --answers answers.jsonl \
  --traces traces.jsonl \
  --output-dir runs/my-agent
```

**Trace submission is optional.** Omit `--traces` to evaluate final answers alone. Without a trace, criteria requiring evidence of intermediate calculations, source checks, or tool use may not be satisfied, resulting in a lower score. There is no fixed penalty for omitting a trace; scoring follows the rubric and the evidence provided.

`--answers` supplies the final answers; `--traces` supplies the assistant messages, tool calls, and tool results used during execution. The grader joins available traces by **`query_id` and `answer_id`** and evaluates each answer together with any submitted trace. See the ready-to-use [answers](examples/answers.jsonl) and [traces](examples/traces.jsonl) examples.

This command runs all three judges and aggregates their results automatically. It makes paid judge calls. Add `--dry-run` to prepare without API calls, then replace it with `--resume` to start. The same `--resume` flag continues interrupted runs using completed batches. `--workers` sets concurrency per judge (default: 4, up to 12 answer jobs across the default panel). See [evaluation details](docs/evaluation.md) for configuration and resume behavior.

## Input formats

**Dataset JSONL** — one question and its weighted rubric per line. This synthetic example is included in [examples/dataset.jsonl](examples/dataset.jsonl):

```json
{
  "query_id": "example_001",
  "query": "A fictional company reports revenue of USD 100 million and operating profit of USD 20 million for 2025. Calculate its operating margin and show the calculation.",
  "information_cutoff": "2025-12-31",
  "rubric": [
    {"criterion_id": "R001", "points": 60.0, "rubric_text": "States that the operating margin is 20%."},
    {"criterion_id": "R002", "points": 40.0, "rubric_text": "Calculates operating margin as operating profit divided by revenue: 20 / 100 × 100 = 20%."}
  ]
}
```

`information_cutoff` is the **query date**, in `YYYY-MM-DD` format. Answers should reflect information available as of that date. The benchmark uses `FAR_` query IDs; retain those IDs when supplying answers.

**Answers JSONL (`--answers`)** — one final answer per query and system, as in [examples/answers.jsonl](examples/answers.jsonl):

```json
{"query_id": "example_001", "answer_id": "my-agent", "answer": "The operating margin is 20%: USD 20 million / USD 100 million × 100 = 20%."}
```

**Traces JSONL (`--traces`, optional)** — one execution trace per query and system for which a trace is available, as in [examples/traces.jsonl](examples/traces.jsonl). This record is expanded for readability; store each record on one line:

```json
{
  "query_id": "example_001",
  "answer_id": "my-agent",
  "trace": [
    {
      "step": 1,
      "assistant_text": "Calculate operating profit divided by revenue, then multiply by 100.",
      "tool_calls": [{"id": "call-1", "name": "calculator", "input": {"expression": "20 / 100 * 100"}}],
      "tool_results": [{"tool_use_id": "call-1", "content": "20", "is_error": false}]
    }
  ]
}
```

Final answers are required; traces are optional and may cover only some answers. Every supplied trace must match an answer's `(query_id, answer_id)` pair; row order can differ. Supply the agent's visible steps in execution order, including assistant text, tool calls, and tool results. Omit unavailable traces; `trace: []` is also accepted for an empty trace. Duplicate, unmatched, or malformed submitted traces are rejected before judge calls. Structured responses may also include sources and limitations; see [supported formats and trace limits](docs/evaluation.md#inputs).

Each `answer_id` identifies a system and must cover every input query exactly once. Multiple systems can share a file. To evaluate a subset, provide matching subsets of the dataset and submissions.

## How judging works

Each judge receives the question, query date, final answer, rendered execution trace, and criterion text. The grader places the final answer before the trace, caps tool arguments at 1,500 characters and tool results at 4,000, and retains the first and last 75,000 characters of an oversized trace with an omission marker. The complete submitted trace and the exact rendered judge input are saved together. Criteria are shuffled, assigned opaque IDs, and evaluated in identical batches of at most 30. Weights and source labels are withheld. Each criterion receives a boolean verdict and a short rationale, using the facts and acceptance rules in the criterion itself.

A criterion passes when at least **two of the three judges** pass it. A question earns the sum of the points assigned to passing criteria. The overall score is the arithmetic mean across questions, on a **0–100** scale. If any judge is incomplete, the ensemble score is withheld until the run is resumed and all members finish.

| Output | Contents |
| --- | --- |
| `scores.json` | Final majority-vote scores and an aggregate for each answer system |
| `summary.json` | Completion status and remaining jobs for each judge |
| `members/<judge>/frozen/answers/` | Submitted answers and agent traces, plus the rendered judge inputs |
| `members/<judge>/traces/` | Judge API call records produced during grading |
| `members/<judge>/` | Each judge's scores, verdicts, and other frozen inputs |
| `ensemble/` | Combined criterion verdicts with individual votes |

### Existing judge runs

`evaluate` handles the panel automatically. You can also combine separately completed judge runs with matching inputs:

```bash
fin-auto-rubric merge \
  --runs runs/judge-a runs/judge-b runs/judge-c \
  --output-dir runs/majority
```

Recompute scores from saved verdicts without model calls with `fin-auto-rubric score --run-dir runs/my-agent`.

## Tests

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

These tests require no API keys or model calls. New grading runs produce results for the supplied inputs and judge configuration.

## Citation

If you use FinAutoRubric or its benchmark, please cite the paper:

```bibtex
@misc{lee2026finautorubric,
  title = {{FinAutoRubric}: Expert-Guided Automatic Rubric Generation for Evaluating Financial Research Agents},
  author = {Hoyoung Lee and Suyeol Yun and Jack Haverty and Yunju Cho and Meesong Kim and Daekyung Park and Sumin Kim and Jihoon Kwon and Jasmine Jia Geng and Andrew Chin and Yin Luo and Edward Tong and Yu Yu and Zach Golkhou and Minkyu Kim and Igor Halperin and Young Cha and Alejandro Lopez-Lira and Chanyeol Choi and Yongjae Lee},
  year = {2026},
  eprint = {2609.35744},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  url = {https://arxiv.org/abs/2609.35744}
}
```

## License

Code is licensed under [Apache 2.0](LICENSE). The benchmark is licensed separately under [CC BY 4.0](https://huggingface.co/datasets/Linq-AI-Research/FinAutoRubric); referenced third-party materials retain their own terms.

Maintained by LinqAlpha. Contact: [hoyounglee@linqalpha.com](mailto:hoyounglee@linqalpha.com).
