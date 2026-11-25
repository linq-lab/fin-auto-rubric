import asyncio
import copy
import json
from pathlib import Path

import pytest

from finautorubric.cli import main
from finautorubric.evaluation import grade
from finautorubric.evaluation.common.backends import AgentResult, BackendError
from finautorubric.evaluation.common.io import read_json, read_jsonl
from finautorubric.evaluation.ensemble import merge
from finautorubric.evaluation.public import evaluate, validate_inputs
from finautorubric.evaluation.scoring import score_run


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


@pytest.fixture
def inputs(tmp_path):
    row = {"query_id": "FAR_example", "query": "Compute a margin", "information_cutoff": "2025-12-31",
           "rubric": [{"criterion_id": "R001:T0001", "points": 60, "rubric_text": "Correct  margin."},
                      {"criterion_id": "R002", "points": 40, "rubric_text": "Shows calculation."}]}
    dataset, answers, config = [tmp_path / name for name in ('dataset.jsonl', 'answers.jsonl', 'judge.toml')]
    write_rows(dataset, [row])
    write_rows(answers, [{"query_id": row['query_id'], "answer_id": "agent", "answer": "My answer", "trace": []}])
    config.write_text('name = "judge-a"\nmodel = "test/model"\nreasoning_effort = "medium"\n')
    return dataset, answers, config, row


def command(inputs, output, *extra):
    dataset, answers, config, _ = inputs
    return ['--dataset', str(dataset), '--answers', str(answers), '--judge-config', str(config),
            '--output-dir', str(output), *extra]


class FakeJudge:
    def __init__(self, passing_text='Correct', fail=False):
        self.calls = []
        self.passing_text = passing_text
        self.fail = fail

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise BackendError('synthetic failure')
        payload = json.loads(kwargs['prompt'].split('```json\n', 1)[1].rsplit('\n```', 1)[0])
        # The answering system, weights, and original rubric IDs are hidden.
        assert set(payload['evaluation_input']) == {'query_id', 'query', 'query_date'}
        assert all(set(c) == {'criterion_id', 'rubric_text'} for c in payload['criteria'])
        assert all(c['criterion_id'].startswith('C') for c in payload['criteria'])
        return AgentResult(output={'verdicts': [{'criterion_id': c['criterion_id'],
                           'pass': self.passing_text in c['rubric_text'], 'rationale': 'Synthetic verdict.'}
                           for c in payload['criteria']]}, metadata={'fixture': True})


def test_public_grading_score_and_explicit_resume(inputs, tmp_path, monkeypatch):
    out = tmp_path / 'run'
    backend = FakeJudge()
    monkeypatch.setattr(grade, 'make_judge', lambda *a, **kw: backend)
    assert evaluate(command(inputs, out, '--dry-run')) == 0
    assert not backend.calls
    key = read_jsonl(out / 'frozen/blind_key.jsonl')[0]['criteria']
    assert {r['original_criterion_id'] for r in key} == {'R001:T0001', 'R002'}
    assert 'Correct  margin.' in {r['rubric_text'] for r in key}
    with pytest.raises(ValueError, match='--resume'):
        evaluate(command(inputs, out))
    assert evaluate(command(inputs, out, '--resume')) == 0
    report = read_json(out / 'scores.json')
    assert report['aggregate'][0]['mean_score'] == 60
    assert report['complete'] is True
    assert len(backend.calls) == 1
    assert evaluate(command(inputs, out, '--resume')) == 0
    assert len(backend.calls) == 1


@pytest.mark.parametrize('change', ['answer', 'config', 'frozen', 'rubric'])
def test_resume_rejects_changed_inputs(inputs, tmp_path, change):
    out = tmp_path / 'run'
    evaluate(command(inputs, out, '--dry-run'))
    dataset, answers, config, row = inputs
    if change == 'answer':
        write_rows(answers, [{'query_id': row['query_id'], 'answer_id': 'agent', 'answer': 'Changed', 'trace': []}])
    elif change == 'config':
        config.write_text(config.read_text().replace('medium', 'high'))
    elif change == 'rubric':
        row['rubric'][0]['rubric_text'] = 'Different fact'
        write_rows(dataset, [row])
    else:
        (out / 'frozen/prompts/evaluator.md').write_text('modified')
    with pytest.raises(ValueError, match='changed'):
        evaluate(command(inputs, out, '--dry-run', '--resume'))


@pytest.mark.parametrize('bad', ['duplicate', 'missing_answer', 'extra_answer', 'unsafe_id', 'date', 'points', 'nan', 'text', 'duplicate_criterion'])
def test_malformed_inputs_fail_before_preparation(inputs, tmp_path, bad):
    dataset, answers, config, row = inputs
    if bad == 'duplicate':
        write_rows(dataset, [row, row])
    elif bad in {'missing_answer', 'extra_answer'}:
        write_rows(answers, [] if bad == 'missing_answer' else [{'query_id': 'unknown', 'answer': 'answer', 'trace': []}])
    else:
        if bad == 'unsafe_id': row['query_id'] = '../escape'
        if bad == 'date': row['information_cutoff'] = '2025-12-31 00:00:00'
        if bad == 'points': row['rubric'][0]['points'] = 59
        if bad == 'nan': row['rubric'][0]['points'] = float('nan')
        if bad == 'text': row['rubric'][0]['rubric_text'] = ''
        if bad == 'duplicate_criterion': row['rubric'][1]['criterion_id'] = row['rubric'][0]['criterion_id']
        write_rows(dataset, [row])
    out = tmp_path / 'bad-run'
    with pytest.raises(ValueError):
        evaluate(command(inputs, out, '--dry-run'))
    assert not out.exists()


def test_separate_queries_and_rubrics_are_directly_evaluable(inputs, tmp_path, monkeypatch):
    dataset, answers, config, row = inputs
    queries, rubrics = tmp_path / 'queries.jsonl', tmp_path / 'public_rubric.jsonl'
    write_rows(queries, [{'query_id': row['query_id'], 'raw_query': row['query'],
                         'information_cutoff': row['information_cutoff'], 'cutoff_boundary': 'inclusive'}])
    write_rows(rubrics, [{k: row[k] for k in ['query_id', 'query', 'rubric']}])
    backend = FakeJudge()
    monkeypatch.setattr(grade, 'make_judge', lambda *a, **kw: backend)
    assert evaluate(['--queries', str(queries), '--rubrics', str(rubrics), '--answers', str(answers),
                     '--judge-config', str(config), '--output-dir', str(tmp_path / 'separate-inputs')]) == 0
    assert len(backend.calls) == 1


def test_failure_does_not_create_a_complete_score(inputs, tmp_path, monkeypatch):
    backend = FakeJudge(fail=True)
    monkeypatch.setattr(grade, 'make_judge', lambda *a, **kw: backend)
    out = tmp_path / 'failed'
    assert evaluate(command(inputs, out)) == 2
    report = read_json(out / 'scores.json')
    assert report['complete'] is False
    assert report['aggregate'][0]['mean_score'] is None
    assert report['aggregate'][0]['questions_scored'] == 0
    assert len(report['missing_jobs']) == 1


def test_blinding_splits_large_rubric_without_losing_criteria(inputs, tmp_path, monkeypatch):
    dataset, _, _, row = inputs
    row['rubric'] = [{'criterion_id': f'R{i}', 'points': 100/61, 'rubric_text': 'Correct'} for i in range(61)]
    write_rows(dataset, [row])
    backend = FakeJudge()
    monkeypatch.setattr(grade, 'make_judge', lambda *a, **kw: backend)
    out = tmp_path / 'large'
    assert evaluate(command(inputs, out)) == 0
    assert len(backend.calls) == 3
    assert read_json(out / 'scores.json')['aggregate'][0]['mean_score'] == pytest.approx(100)


def test_majority_vote_preserves_weights_and_checks_answers(inputs, tmp_path, monkeypatch):
    dataset, answers, config, row = inputs
    members = []
    for number, passing in enumerate(['Correct', 'Correct', 'calculation']):
        config.write_text(f'name = "judge-{number}"\nmodel = "test/model-{number}"\n')
        backend = FakeJudge(passing_text=passing)
        monkeypatch.setattr(grade, 'make_judge', lambda *a, **kw: backend)
        out = tmp_path / f'judge-{number}'
        assert evaluate(command(inputs, out)) == 0
        members.append(out)
    output = tmp_path / 'majority'
    merge(members, output)
    assert score_run(output)['aggregate'][0]['mean_score'] == 60
    with pytest.raises(ValueError, match='empty'):
        merge(members, output)
    answer_path = next((members[0] / 'frozen/answers').rglob('*.json'))
    answer_path.write_text('{}')
    with pytest.raises(ValueError, match='answers'):
        merge(members, tmp_path / 'bad-merge')


def test_cli_errors_are_actionable(inputs, tmp_path):
    assert main(['evaluate', *command(inputs, tmp_path / 'run', '--dry-run')]) == 0
    assert main(['evaluate', *command(inputs, tmp_path / 'run', '--dry-run')]) == 2
