import asyncio
from pathlib import Path

import pytest

from finautorubric.evaluation import grade
from finautorubric.evaluation.common.io import read_json
from finautorubric.evaluation.public import DEFAULT_CONFIG, evaluate, load_config
from finautorubric.evaluation.scoring import score_run
from .test_public_evaluation import FakeJudge, inputs


def panel_command(inputs, out, *extra):
    dataset, answers, _, _ = inputs
    return ['--dataset', str(dataset), '--answers', str(answers), '--output-dir', str(out), *extra]


def test_default_models_and_routes_match_research_panel():
    configs = load_config()
    assert [c['model'] for c in configs] == [
        'meta/muse-spark-1.3-contributor', 'z-ai/glm-5.3-flash', 'deepseek/deepseek-v4.1-flash']
    assert all(c['reasoning_effort'] == 'medium' for c in configs)
    assert configs[1]['provider'] == {'require_parameters': True, 'quantizations': ['fp8', 'bf16'], 'ignore': ['phala']}
    assert configs[2]['provider'] == {'require_parameters': True, 'quantizations': ['fp8', 'bf16']}
    assert load_config(Path(__file__).resolve().parents[1] / 'examples/judge.toml') == configs


def test_default_panel_prepares_without_credentials_from_any_directory(inputs, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    monkeypatch.setattr(grade, 'make_judge', lambda *a, **k: pytest.fail('dry-run contacted a model'))
    out = tmp_path / 'panel'
    assert evaluate(panel_command(inputs, out, '--dry-run')) == 0
    manifest = read_json(out / 'manifest.json')
    assert len(manifest['judges']) == 3
    assert all((out / p / 'frozen/blind_key.jsonl').is_file() for p in manifest['members'])
    assert not (out / 'ensemble').exists()
    assert score_run(out)['complete'] is False


def test_default_panel_runs_concurrently_and_merges_automatically(inputs, tmp_path, monkeypatch):
    names = [c['name'] for c in load_config()]
    started = set()
    barrier = asyncio.Event()

    class ConcurrentJudge(FakeJudge):
        def __init__(self, name, passing):
            super().__init__(passing_text=passing)
            self.name = name

        async def generate(self, **kwargs):
            started.add(self.name)
            if len(started) == 3:
                barrier.set()
            await asyncio.wait_for(barrier.wait(), timeout=2)
            return await super().generate(**kwargs)

    backends = {name: ConcurrentJudge(name, 'Correct' if index < 2 else 'calculation')
                for index, name in enumerate(names)}
    monkeypatch.setattr(grade, 'make_judge', lambda name, **kw: backends[name])
    out = tmp_path / 'panel'
    assert evaluate(panel_command(inputs, out)) == 0
    assert started == set(names)
    assert all(len(b.calls) == 1 for b in backends.values())
    scores = read_json(out / 'scores.json')
    assert scores['complete'] and scores['aggregate'][0]['mean_score'] == 60
    assert read_json(out / 'summary.json')['completed_judges'] == 3
    verdict_file = next((out / 'ensemble/grades').rglob('*.json'))
    verdict = read_json(verdict_file)
    assert all(set(v['votes']) == set(names) for v in verdict['verdicts'])
    original = verdict_file.read_bytes()
    monkeypatch.setattr(grade, 'make_judge', lambda *a, **k: pytest.fail('completed run reopened a backend'))
    assert evaluate(panel_command(inputs, out, '--resume')) == 0
    assert verdict_file.read_bytes() == original
    assert score_run(out)['aggregate'][0]['mean_score'] == 60


def test_failed_judge_withholds_ensemble_and_resume_only_calls_missing_member(inputs, tmp_path, monkeypatch):
    names = [c['name'] for c in load_config()]
    backends = {name: FakeJudge(fail=index == 2) for index, name in enumerate(names)}
    opened = []
    def backend(name, **kw):
        opened.append(name)
        return backends[name]
    monkeypatch.setattr(grade, 'make_judge', backend)
    out = tmp_path / 'panel'
    assert evaluate(panel_command(inputs, out)) == 2
    assert not (out / 'ensemble').exists()
    assert read_json(out / 'summary.json')['completed_judges'] == 2
    assert read_json(out / 'scores.json')['aggregate'] == []
    assert score_run(out)['missing_members'] == [names[2]]
    backends[names[2]].fail = False
    opened.clear()
    assert evaluate(panel_command(inputs, out, '--resume')) == 0
    assert opened == [names[2]]
    assert [len(backends[n].calls) for n in names] == [1, 1, 2]
    assert read_json(out / 'scores.json')['complete']


def test_all_frozen_members_are_checked_before_any_paid_call(inputs, tmp_path, monkeypatch):
    out = tmp_path / 'panel'
    assert evaluate(panel_command(inputs, out, '--dry-run')) == 0
    name = load_config()[1]['name']
    (out / 'members' / name / 'frozen/prompts/evaluator.md').write_text('changed')
    monkeypatch.setattr(grade, 'make_judge', lambda *a, **k: pytest.fail('called model before checking every member'))
    with pytest.raises(ValueError, match='Frozen input changed'):
        evaluate(panel_command(inputs, out, '--resume'))


@pytest.mark.parametrize('kind', ['empty', 'even', 'duplicate_name', 'duplicate_model'])
def test_invalid_panels_are_rejected(inputs, tmp_path, kind):
    config = tmp_path / 'panel.toml'
    entries = [('a', 'test/a'), ('b', 'test/b'), ('c', 'test/c')]
    if kind == 'empty': entries = []
    if kind == 'even': entries = entries[:2]
    if kind == 'duplicate_name': entries[1] = ('a', 'test/b')
    if kind == 'duplicate_model': entries[1] = ('b', 'test/a')
    config.write_text(''.join(f'[[judges]]\nname = "{name}"\nmodel = "{model}"\n' for name, model in entries)
                      or 'judges = []\n')
    out = tmp_path / 'bad'
    with pytest.raises(ValueError):
        evaluate(panel_command(inputs, out, '--judge-config', str(config), '--dry-run'))
    assert not out.exists()


def test_custom_models_and_single_judge_override(inputs, tmp_path, monkeypatch):
    config = tmp_path / 'custom.toml'
    config.write_text('[[judges]]\nname = "custom"\nmodel = "custom/model"\nreasoning_effort = "low"\n')
    seen = []
    def backend(name, **kw):
        seen.append(grade.JUDGE_SPECS[name])
        return FakeJudge()
    monkeypatch.setattr(grade, 'make_judge', backend)
    out = tmp_path / 'single'
    assert evaluate(panel_command(inputs, out, '--judge-config', str(config))) == 0
    assert seen[0]['model'] == 'custom/model' and seen[0]['reasoning_effort'] == 'low'
    assert not (out / 'members').exists()
    assert read_json(out / 'scores.json')['aggregate'][0]['mean_score'] == 60


def test_changed_panel_config_cannot_resume_existing_results(inputs, tmp_path):
    config = tmp_path / 'panel.toml'
    config.write_text(DEFAULT_CONFIG.read_text())
    out = tmp_path / 'panel'
    assert evaluate(panel_command(inputs, out, '--judge-config', str(config), '--dry-run')) == 0
    config.write_text(config.read_text().replace('deepseek/deepseek-v4.1-flash', 'custom/different-model'))
    with pytest.raises(ValueError, match='configuration changed'):
        evaluate(panel_command(inputs, out, '--judge-config', str(config), '--dry-run', '--resume'))


def test_all_three_models_can_be_replaced(inputs, tmp_path, monkeypatch):
    config = tmp_path / 'custom-panel.toml'
    config.write_text(''.join(f'[[judges]]\nname = "custom-{i}"\nmodel = "custom/model-{i}"\nreasoning_effort = "low"\n'
                              for i in range(3)))
    observed = []
    def backend(name, **kw):
        observed.append(grade.JUDGE_SPECS[name])
        return FakeJudge()
    monkeypatch.setattr(grade, 'make_judge', backend)
    out = tmp_path / 'custom-panel'
    assert evaluate(panel_command(inputs, out, '--judge-config', str(config))) == 0
    assert {s['model'] for s in observed} == {f'custom/model-{i}' for i in range(3)}
    assert all(s['reasoning_effort'] == 'low' for s in observed)
    assert read_json(out / 'scores.json')['complete']


def test_changed_ensemble_files_are_not_reused(inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(grade, 'make_judge', lambda *a, **kw: FakeJudge())
    out = tmp_path / 'panel'
    assert evaluate(panel_command(inputs, out)) == 0
    path = out / 'ensemble/frozen/blind_key.jsonl'
    path.write_text(path.read_text() + '\n')
    monkeypatch.setattr(grade, 'make_judge', lambda *a, **k: pytest.fail('reopened a backend'))
    with pytest.raises(ValueError, match='Saved ensemble or member grades changed'):
        evaluate(panel_command(inputs, out, '--resume'))
