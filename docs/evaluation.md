# Evaluation formats and behavior

The grader accepts either the combined Hugging Face JSONL (`--dataset`) or separate question and rubric files (`--queries`, `--rubrics`). It evaluates the embedded criterion text with the released weights. Internal research bundles requiring a `scoring_policy.json` sidecar are rejected.

## Inputs

Queries must have unique `query_id` values, nonempty `query` text (`raw_query` is also accepted), and valid date-only `information_cutoff` values. Every query needs one rubric, with unique criterion IDs, positive finite points, and a 100-point total. IDs used in file paths must start with a letter or digit and contain only letters, digits, underscores, dots, or hyphens.

Pass final answers with `--answers answers.jsonl` and optionally include execution traces with `--traces traces.jsonl`. Traces may cover all, some, or none of the answers. Each submitted trace must match exactly one `(query_id, answer_id)` pair in the answer file; line order does not need to match.

**Traces are optional, but omitting them may lower scores.** If the final answer alone does not demonstrate required calculations, source checks, or tool use, the corresponding rubric criteria may fail. There is no fixed deduction for a missing trace: judges assess only the evidence provided and do not infer missing work.

`answers.jsonl` contains nonempty final `answer` strings:

```json
{"query_id": "example_001", "answer_id": "my-agent", "answer": "The operating margin is 20%."}
```

`traces.jsonl` contains the corresponding `trace` lists:

```json
{"query_id": "example_001", "answer_id": "my-agent", "trace": [{"assistant_text": "20 / 100 × 100 = 20%."}]}
```

The answer file also accepts a structured response, paired with the same trace file:

```json
{"query_id": "example_001", "answer_id": "my-agent", "response": {"answer": "The operating margin is 20%.", "sources": [], "limitations": "Based on the figures in the question."}}
```

`payload` is accepted as an alias for `response`. Supply exactly one of `answer`, `response`, or `payload`. Optional `sources` and `limitations` can be placed beside the answer or inside the structured response, but each field must appear only once. The grader joins the answer and trace by both IDs, then combines them into the judge's `candidate_response.answer`; `sources` and `limitations` are passed alongside it.

The earlier combined-file format is also supported: omit `--traces` and put `trace` beside each answer or inside its structured response. When `--traces` is supplied, inline traces are rejected to avoid ambiguous inputs.

Every system needs an answer for every supplied query. Duplicate, missing, or extra query/answer pairs, and duplicate, unmatched, or malformed submitted traces are rejected before any API call. Both IDs are required on every row when using separate files. For a partial benchmark, supply matching dataset and answer subsets, plus any available traces for those answers; the score then describes that subset. Omit unavailable traces; an explicit `trace: []` is also accepted. A supplied trace record must still contain a valid `trace` list.

### Trace steps

`trace` is an ordered array of visible execution steps. Each step must include at least one nonempty assistant message, tool call, or tool result. Tool calls are recorded data: the grader does not execute them. Adapt your agent's logs to these fields:

| Field | Format |
| --- | --- |
| `step` | Optional nonnegative integer; defaults to the 1-based array position. Step numbers must increase in array order. |
| `assistant_text` | String containing the visible assistant message; defaults to `""`. |
| `tool_calls` | Array of objects with nonempty string `id`, nonempty string `name`, and JSON object `input`; defaults to `[]`. |
| `tool_results` | Array of objects with nonempty string `tool_use_id`, string `content`, and optional boolean `is_error` (default `false`); defaults to `[]`. |

Use matching call/result IDs. A result can appear in a later step. Additional step metadata is preserved in the frozen submission but is not shown to the judge. Provider-specific `messages`, raw trace file paths, and whole harness run objects must be converted to this step array. All numbers must be finite JSON values.

### Trace rendering and saved inputs

The grader renders the final answer first, under `AGENT'S FINAL ANSWER:`, followed by `AGENT'S TRACE (assistant text, tool calls, tool results):`. An omitted trace is marked `[trace not provided]`; an explicitly empty trace is marked `[no trace steps]`. Within each step, it renders assistant text, tool calls, then tool results in their submitted order. Tool argument object keys are sorted for deterministic rendering.

The rendering policy uses the limits from the research BFB trace renderer: 1,500 characters per tool-call argument object, 4,000 per tool result, and a 150,000-character trace budget. Per-item truncation adds `...`; whole-trace overflow keeps the first and last 75,000 characters and inserts an omission marker between them. Markers are additional to these limits. The final answer is not truncated. This matches the research layout and limits; canonical argument ordering means it is not a byte-for-byte reproduction of every historical input.

Each `frozen/answers/answers/<answer_id>/<query_id>.json` stores `response` (the original answer and complete trace) and `candidate_response` (the exact response shown to the judge), plus source file paths. In a panel, this path is under each `members/<judge>/` directory. Both are frozen during `--dry-run`, and rendering policy/version is recorded in `request.json` and member manifests. The separate trace file's SHA-256 is included in `request.json` and checked on resume. The `traces/` output directory contains **judge API call records**, separate from the submitted agent traces.

## Judge configuration

Omitting `--judge-config` selects the packaged panel: `meta/muse-spark-1.3-contributor`, `z-ai/glm-5.3-flash`, and `deepseek/deepseek-v4.1-flash`, all at `medium` reasoning effort. The same defaults are in [examples/judge.toml](../examples/judge.toml). These are the common three judges used in the evaluation experiments; the earlier DeepSeek V4 snapshot is not selected.

To customize the panel, edit its `[[judges]]` tables and pass `--judge-config PATH`. Each entry has a unique `name`, a distinct OpenRouter `model`, a `reasoning_effort`, and optional token, timeout, retry, and `[judges.provider]` settings. Use one judge or an odd number of at least three judges. A flat single-judge TOML is also supported; [examples/judge-single.toml](../examples/judge-single.toml) provides one.

The backend calls OpenRouter's chat-completions API with a strict JSON schema and the requested reasoning effort. Each provider table is passed as routing configuration. The defaults retain fp8/bf16 routing for GLM and DeepSeek and exclude Phala for GLM, as in the recorded experiment configuration. The judges receive no web tools: criterion text supplies the facts and acceptance rules.

`request_timeout_seconds` bounds each HTTP attempt. `max_retries` bounds HTTP attempts within the per-stage `--timeout-seconds` budget. Invalid verdict structures may trigger up to three validation attempts; a capacity error permits one additional capacity retry. These retries can incur charges. Judges execute concurrently, and `--workers` controls concurrent answer jobs **per judge**. The default of 4 gives up to 12 answer jobs across three judges. Batches are sequential within each job.

## Offline preparation and resume

`--dry-run` validates and freezes every member's inputs before any model call. Replace it with `--resume` to begin evaluation or continue an interrupted run. Successful batches are reused, and completed members do not reopen their backends. Incomplete work is attempted again only when you run the command. There is no background scheduler.

A changed dataset, rubric, answer or agent trace, rendering policy, judge configuration, seed, or packaged judge prompt requires a fresh output directory. Frozen submissions and rendered inputs are checked for modification on resume. Answer-only runs prepared before the answer-and-trace input contract require a fresh directory. Logs and traces remain under the chosen run directory.

For example, prepare the benchmark evaluation with the default panel and optional traces (omit `--traces traces.jsonl` if unavailable):

```bash
fin-auto-rubric evaluate \
  --dataset data/finautorubric_public.jsonl \
  --answers answers.jsonl \
  --traces traces.jsonl \
  --output-dir runs/my-agent \
  --dry-run
```

Then start it with the same inputs and configuration:

```bash
fin-auto-rubric evaluate \
  --dataset data/finautorubric_public.jsonl \
  --answers answers.jsonl \
  --traces traces.jsonl \
  --output-dir runs/my-agent \
  --resume
```

## Scores and majority voting

Each majority-passing criterion earns its assigned points. Every question totals 100 points, so the overall score is the arithmetic mean of question scores. `scores.json` at the run root contains the final per-question scores and one aggregate per `answer_id`.

An ensemble requires an odd number of at least three distinct judges. Members use identical questions, answers, rubric weights, criterion ordering, and judge prompts. A criterion passes when more than half of the judges pass it. A failed member is not dropped from the vote: the other members finish, their artifacts remain available, and the ensemble score is withheld until every member completes.

Panel runs store each member under `members/<judge>/` and the combined verdicts, including individual votes, under `ensemble/`. `score --run-dir RUN` recomputes the final scores without model calls. Already completed ensembles are reused on resume after checking their saved artifacts. A single-judge configuration retains the flat `grades/`, `frozen/`, and `traces/` layout; incomplete single-judge scores have `mean_score: null`.

The standalone `merge` command can combine separately completed runs into a new directory while preserving the member runs. The default `evaluate` command performs this step automatically.

The grader preserves the research evaluator prompt and batching rules. This default uses criterion-level majority voting; paper analyses that average judge scores or substitute a non-self judge use different aggregation settings. Matching these three model names alone does not reproduce every paper table.
