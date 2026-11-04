"""Regression coverage for the agentic-benchmark PR review."""
import hashlib
import json
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from mlsurgie import Budget, Feature, Goal, ScriptedEditor, SelectionRule, Table, Track, build, replay
from mlsurgie.data import Preprocessor
from mlsurgie.metrics import silhouette
from mlsurgie.network import apply_program, rebase_numeric_program

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = {"base": "zero", "hidden_sizes": [], "rationale": "Regression fixture", "operations": []}


def test_benchmark_data_checksums():
    root = ROOT / 'benchmarks/data'
    checksums = json.loads((root / 'checksums.json').read_text())
    assert set(checksums) == {'fraud.csv', 'segments.csv', 'wine.csv', 'diabetes.csv', 'breast_cancer.csv'}
    for name, expected in checksums.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected


def test_anonymized_python_api_and_config_hide_semantics(tmp_path):
    from mlsurgie.config import load_task
    from mlsurgie.tracks import anonymize
    rows = [{'secret_amount': i, 'merchant': f'merchant_{i % 12:02d}', 'fraud': str(i % 2)} for i in range(120)]
    table = Table(rows, [Feature('secret_amount', description='dollars'), Feature('merchant', 'categorical')],
                  Goal('Detect fraud', 'fraud'))
    transformed = anonymize(table)
    # Alias sort order must preserve the feature basis even with >=10 categories.
    original_pre = Preprocessor.from_support(rows, table.features)
    anonymous_pre = Preprocessor.from_support(transformed.rows, transformed.features)
    np.testing.assert_array_equal(original_pre.transform(rows), anonymous_pre.transform(transformed.rows))
    seen = []
    class Spy(ScriptedEditor):
        def propose(self, context, timeout):
            seen.append(context)
            return super().propose(context, timeout)
    build(table=table, editor=Spy([PROGRAM]), output=tmp_path / 'api', budget=Budget(iterations=1),
          track=Track(names='anonymized'))
    import csv
    source = tmp_path / 'source.csv'
    with source.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = tmp_path / 'task.json'
    config.write_text(json.dumps({'dataset': 'source.csv', 'goal': table.goal.to_dict(),
                                  'features': [f.__dict__ for f in table.features], 'budget': {'iterations': 1}}))
    loaded, options, _ = load_task(config, track=Track(names='anonymized'))
    build(table=loaded, editor=Spy([PROGRAM]), output=tmp_path / 'config', **options)
    assert seen[0]['encoded_features'] == seen[1]['encoded_features']
    for context in seen:
        text = json.dumps(context)
        assert not any(secret in text for secret in ('secret_amount', 'merchant', 'fraud', 'dollars'))


def test_historical_model_rebase_preserves_outputs_and_replays(tmp_path):
    historical_rows = [{'x': None}, {'x': 10}, {'x': 20}]
    current_rows = [{'x': None}, {'x': -5}, {'x': 100}]
    feature = [Feature('x')]
    old = Preprocessor.from_support(historical_rows, feature)
    new = Preprocessor.from_support(current_rows, feature)
    program = {**PROGRAM, 'hidden_sizes': [2], 'activations': ['tanh'], 'operations': [
        {'op': 'set', 'tensor': 'layers.0.weight', 'value': [[2, 3], [-1, 4]]},
        {'op': 'set', 'tensor': 'layers.0.bias', 'value': [0.2, -0.4]},
        {'op': 'set', 'tensor': 'layers.1.weight', 'value': [[3, 2]]}]}
    budget = Budget(iterations=1)
    before = apply_program(program, 2, budget)
    converted = rebase_numeric_program(program, old.state, new.state, budget)
    after = apply_program(converted, 2, budget)
    probe = [{'x': None}, {'x': -12}, {'x': 10}, {'x': 25}]
    np.testing.assert_allclose(before.outputs(old.transform(probe)), after.outputs(new.transform(probe)), atol=2e-5)
    table = Table([{'x': i, 'y': float(i)} for i in range(60)], feature, Goal('Response', 'y', task='regression'))
    result = build(table=table, editor=ScriptedEditor([]), output=tmp_path / 'run',
                   initial_program=program, initial_preprocessing=old.state, budget=budget, track=Track(names='anonymized'))
    assert replay(result.path).fingerprint() == result.model.network.fingerprint()
    assert result.report['selected_iteration'] == 0


def test_repair_fixture_never_reads_benchmark_rows(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from benchmarks.make_repair import make_fixture
    # Fixture generation must work even when every CSV read is refused.
    original = Path.open
    def guarded(path, *args, **kwargs):
        if path.suffix == '.csv':
            raise AssertionError('Repair fixture accessed benchmark data')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    program, preprocessing = make_fixture()
    assert program == json.loads((ROOT / 'benchmarks/tasks/segments_repair.program.json').read_text())
    assert preprocessing == json.loads((ROOT / 'benchmarks/tasks/segments_repair.preprocessing.json').read_text())


def test_selection_override_reaches_baselines_and_summary(monkeypatch, tmp_path):
    pytest.importorskip('sklearn')
    import mlsurgie.baselines as baselines
    from mlsurgie.suite import run_suite
    class Model:
        def __init__(self, alpha): self.alpha = alpha
        def fit(self, x, y): return self
        def get_params(self): return {'alpha': self.alpha}
    monkeypatch.setattr(baselines, '_candidates', lambda *args: {'ridge': [Model(1), Model(2)]})
    monkeypatch.setattr(baselines, '_metrics', lambda goal, model, x, y:
                        {'rmse': .1 if model.alpha == 1 else .2, 'mae': .4 if model.alpha == 1 else .2, 'r2': .5})
    table = Table([{'x': i, 'target': i} for i in range(60)], [Feature('x')], Goal('Response', 'target', task='regression'))
    assert baselines.run_baselines(table, selection=SelectionRule('mae'))['ridge']['hyperparameters']['alpha'] == 2
    assert baselines.run_baselines(table, selection=SelectionRule('mae', .25))['ridge']['hyperparameters']['alpha'] == 1
    import csv
    with (tmp_path / 'data.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['x', 'target'])
        writer.writeheader()
        writer.writerows(table.rows)
    (tmp_path / 'task.json').write_text(json.dumps({'dataset': 'data.csv', 'goal': table.goal.to_dict(),
        'features': [{'name': 'x'}], 'selection': {'metric': 'mae'}, 'budget': {'iterations': 1}}))
    (tmp_path / 'suite.json').write_text(json.dumps({'tasks': ['task.json'], 'seeds': [0]}))
    summary = run_suite(tmp_path / 'suite.json', lambda *args: ScriptedEditor([PROGRAM]), tmp_path / 'suite')
    cell = summary['cells'][0]
    assert cell['selection_metric'] == summary['table'][0]['headline'] == 'mae'
    assert cell['baselines']['ridge']['hyperparameters']['alpha'] == 2
    assert 'mae ↓' in (tmp_path / 'suite/results.md').read_text()


def test_silhouette_bounds_pairwise_memory_and_matches_reference():
    sklearn = pytest.importorskip('sklearn.metrics')
    x = np.random.default_rng(0).normal(size=(200, 128))
    labels = np.arange(200) % 3
    tracemalloc.start()
    try:
        result = silhouette(x, labels)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 8_000_000  # old [200,200,128] temporary alone exceeded 40 MB
    assert result == pytest.approx(sklearn.silhouette_score(x, labels), abs=1e-12)
