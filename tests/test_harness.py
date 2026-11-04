"""Harness protocol tests use explicit fixtures, never paid model calls."""
import csv
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from mlsurgie import Budget, Feature, Goal, Table, Track, build, load, replay
from mlsurgie.harness import OpenHandsEditor
from mlsurgie.sandbox import DockerSandbox


def data():
    return Table([{'x': i - 50, 'label': str(int(i >= 50))} for i in range(100)],
                 [Feature('x')], Goal('Predict sign by direct weights', 'label'))


def candidate(width):
    return {'base': 'zero', 'hidden_sizes': width, 'rationale': 'Known-rule protocol fixture', 'operations': []}


def sign_program(pre):
    program = candidate([2])
    program['operations'] = [
        {'op': 'set', 'tensor': 'layers.0.weight', 'value': [[1, 0], [-1, 0]]},
        {'op': 'set', 'tensor': 'layers.1.weight', 'value': [[8 * pre['scale'], -8 * pre['scale']]]},
        {'op': 'set', 'tensor': 'layers.1.bias', 'value': [8 * pre['mean']]},
    ]
    return program


class FixtureSandbox:
    """Stands in for Docker: reads the prepared inputs and writes into the persistent workspace."""
    def check(self): pass

    def execute(self, source, *, inputs, skills, workspace, record, timeout, language='python'):
        record.mkdir(parents=True)
        (record / ('source.py' if language == 'python' else 'source.sh')).write_text(source)
        state = json.loads((inputs / 'state.json').read_text())
        task = json.loads((inputs / 'task.json').read_text())
        i = state['iteration']
        program = candidate([] if i == 1 else [3]) if i != 2 else sign_program(task['preprocessing'][0])
        (workspace / 'candidate.json').write_text(json.dumps(program))
        return 0, 'Fixture, not code execution'


def test_harness_prepares_files_revises_topology_and_replays(monkeypatch, tmp_path):
    monkeypatch.setenv('MLSURGIE_TEST_KEY', 'never-write-this-secret')
    contexts = []
    def runtime(editor, context, timeout):
        contexts.append(context)
        assert context['frozen_hidden_sizes'] is None and 'test' not in context
        inputs = editor.inputs
        with (inputs / 'data' / 'support.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == context['support_count'] and set(rows[0]) == {'x', 'label'}
        with (inputs / 'data' / 'support_encoded.csv').open() as handle:
            assert next(csv.reader(handle)) == ['x:standardized', 'x:missing', 'label']
        assert json.loads((inputs / 'state.json').read_text())['iteration'] == context['iteration']
        assert 'captured' in editor.execute('# explicit test fixture')
        return {'input_tokens': 10, 'output_tokens': 20}
    editor = OpenHandsEditor('test/fixture', tmp_path / 'run', 'MLSURGIE_TEST_KEY',
                             sandbox=FixtureSandbox(), runtime=runtime)
    editor.identity['kind'] = 'scripted_fixture'
    result = build(table=data(), editor=editor, output=tmp_path / 'run',
                   budget=Budget(iterations=3), dynamic_architecture=True, isolated_evaluation=True)
    assert result.report['selected_iteration'] == 2
    assert result.report['history'][2]['selected'] is False
    assert result.report['final_test']['accuracy'] == 1
    assert result.report['cost']['editor_tokens'] == 90
    assert len(contexts) == 3
    predictor = load(result.path)
    assert predictor.network.hidden_sizes == [2]
    assert replay(result.path).fingerprint() == predictor.network.fingerprint()
    np.testing.assert_array_equal(predictor.predict([{'x': -1}, {'x': 1}]), ['0', '1'])
    # Development and test rows never reach the agent's files.
    support = {str(r['x']) for r in csv.DictReader((editor.inputs / 'data' / 'support.csv').open())}
    manifest = json.loads((result.path / 'manifest.json').read_text())
    table = data()
    hidden = {str(table.rows[i]['x']) for k in ('development', 'test') for i in manifest['split_indices'][k]}
    assert not support & hidden
    for path in result.path.rglob('*'):
        if path.is_file() and path.suffix in {'.json', '.csv', '.md', '.log'}:
            assert 'never-write-this-secret' not in path.read_text()


def test_brief_is_compact_and_previous_candidate_is_not_resubmitted(monkeypatch, tmp_path):
    monkeypatch.setenv('MLSURGIE_TEST_KEY', 'test')
    seen = []
    def runtime(editor, context, timeout):
        seen.append(editor._brief(context))
        if context['iteration'] == 1:
            editor.execute('# write candidate')
        else:
            assert not (editor.workspace / 'candidate.json').exists()
            assert (editor.workspace / 'previous_candidate.json').exists()
        return {'input_tokens': 1, 'output_tokens': 1}
    table = Table([{'x': i, 'label': str(int(i >= 100))} for i in range(200)], [Feature('x')], Goal('Sign', 'label'))
    editor = OpenHandsEditor('test/model', tmp_path / 'run', 'MLSURGIE_TEST_KEY', sandbox=FixtureSandbox(), runtime=runtime)
    result = build(table=table, editor=editor, output=tmp_path / 'run', budget=Budget(iterations=2, support_examples=200),
                   dynamic_architecture=True)
    assert result.report['history'][1]['status'] == 'invalid'  # no new candidate: invalid, never a stale resubmission
    assert all(len(b) < 6000 for b in seen) and '/skills/README.md' in seen[0]


def test_harness_failed_code_does_not_reuse_old_candidate(monkeypatch, tmp_path):
    monkeypatch.setenv('MLSURGIE_TEST_KEY', 'test')
    class Failing(FixtureSandbox):
        def execute(self, source, **kwargs):
            kwargs['record'].mkdir(parents=True)
            return 1, 'Traceback: failure'
    def runtime(editor, context, timeout):
        editor.pending = candidate([])
        assert 'failed' in editor.execute('bad code')
        return {'input_tokens': 1, 'output_tokens': 1}
    editor = OpenHandsEditor('test/model', tmp_path / 'run', 'MLSURGIE_TEST_KEY', sandbox=Failing(), runtime=runtime)
    result = build(table=data(), editor=editor, output=tmp_path / 'run', budget=Budget(iterations=1),
                   dynamic_architecture=True)
    assert result.model is None
    assert result.report['history'][0]['status'] == 'invalid'


@pytest.mark.parametrize('policy, status', [('strict', 'policy_violation'), ('open', 'evaluated')])
def test_fitting_audit_rejects_on_strict_and_labels_on_open(monkeypatch, tmp_path, policy, status):
    monkeypatch.setenv('MLSURGIE_TEST_KEY', 'test')
    def runtime(editor, context, timeout):
        (editor.workspace / 'tools.py').write_text('import numpy as np\nw = np.linalg.lstsq(X, y, rcond=None)[0]\n')
        editor.execute('from tools import w')
        return {'input_tokens': 1, 'output_tokens': 1}
    editor = OpenHandsEditor('test/model', tmp_path / 'run', 'MLSURGIE_TEST_KEY', sandbox=FixtureSandbox(), runtime=runtime)
    result = build(table=data(), editor=editor, output=tmp_path / 'run', budget=Budget(iterations=1),
                   dynamic_architecture=True, track=Track(policy=policy))
    record = result.report['history'][0]
    assert record['status'] == status
    assert record['audit']['violations'][0]['rule'] == 'closed-form or numerical solver'
    assert record['audit']['violations'][0]['file'].endswith('tools.py')


def test_audit_rules():
    from mlsurgie.audit import scan_text
    flagged = ['from sklearn.linear_model import LogisticRegression', 'import torch', 'w -= lr * g',
               'grad = X.T @ (p - y)', 'from scipy.optimize import minimize', 'model.fit(X, y)',
               'c = np.polyfit(x, y, 1)', 'loss.backward()']
    clean = ['rate = y[mask].mean()', 'w = np.log(p / (1 - p))', 'q = np.quantile(x, [.25, .5])',
             '# np.linalg.lstsq is not allowed', 'counts = df.groupby("c")["y"].agg(["mean", "size"])']
    assert all(scan_text(line, 'f') for line in flagged)
    assert not any(scan_text(line, 'f') for line in clean)


def test_container_mounts_isolation_and_persistent_workspace(monkeypatch, tmp_path):
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr('mlsurgie.sandbox.subprocess.run', run)
    for name in ('inputs', 'skills', 'workspace'):
        (tmp_path / name).mkdir()
    code, _ = DockerSandbox().execute('pass', inputs=tmp_path / 'inputs', skills=tmp_path / 'skills',
                                      workspace=tmp_path / 'workspace', record=tmp_path / 'rec', timeout=12)
    cmd = commands[0]
    assert code == 0
    assert cmd[cmd.index('--network') + 1] == 'none'
    assert '--read-only' in cmd and '--cap-drop' in cmd and '--pids-limit' in cmd and '--memory' in cmd
    envs = [cmd[i + 1] for i, part in enumerate(cmd) if part == '--env']
    assert envs == ['HOME=/tmp', 'MPLCONFIGDIR=/tmp', 'PYTHONPATH=/workspace']
    assert '--privileged' not in cmd and 'docker.sock' not in ' '.join(cmd)
    mounts = [cmd[i + 1] for i, part in enumerate(cmd) if part == '--mount']
    assert any(m.endswith('dst=/inputs,readonly') for m in mounts)
    assert any(m.endswith('dst=/skills,readonly') for m in mounts)
    assert any(m.endswith('dst=/workspace') for m in mounts)
    assert (tmp_path / 'rec' / 'source.py').read_text() == 'pass'
    assert commands[-1][1:3] == ['rm', '-f']


def test_container_timeout_cleans_up(monkeypatch, tmp_path):
    commands = []
    def run(cmd, **kwargs):
        commands.append(cmd)
        if cmd[1] == 'run':
            raise subprocess.TimeoutExpired(cmd, 1)
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr('mlsurgie.sandbox.subprocess.run', run)
    with pytest.raises(ValueError, match='deadline'):
        DockerSandbox().execute('pass', inputs=tmp_path, skills=tmp_path, workspace=tmp_path, record=tmp_path / 'r', timeout=1)
    assert commands[-1][1:3] == ['rm', '-f']


def test_symlinked_candidate_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv('MLSURGIE_TEST_KEY', 'test')
    class Linking(FixtureSandbox):
        def execute(self, source, **kwargs):
            kwargs['record'].mkdir(parents=True)
            (kwargs['workspace'] / 'candidate.json').symlink_to(kwargs['inputs'] / 'task.json')
            return 0, ''
    def runtime(editor, context, timeout):
        assert 'regular file' in editor.execute('x')
        return {'input_tokens': 1, 'output_tokens': 1}
    editor = OpenHandsEditor('test/model', tmp_path / 'run', 'MLSURGIE_TEST_KEY', sandbox=Linking(), runtime=runtime)
    result = build(table=data(), editor=editor, output=tmp_path / 'run', budget=Budget(iterations=1), dynamic_architecture=True)
    assert result.report['history'][0]['status'] == 'invalid'


def test_sdk_tool_initialization_without_model_call(monkeypatch, tmp_path):
    monkeypatch.setenv('OH_PERSISTENCE_DIR', str(tmp_path / 'sdk-state'))
    monkeypatch.setenv('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
    monkeypatch.setenv('OPENHANDS_SUPPRESS_BANNER', '1')
    pytest.importorskip('openhands.sdk')
    monkeypatch.setenv('MLSURGIE_TEST_KEY', 'test-key')
    editor = OpenHandsEditor('openrouter/test/model', tmp_path, 'MLSURGIE_TEST_KEY')
    editor.context = {'output_token_limit': 200, 'limits': {'max_editor_tokens': 1000}}
    editor.deadline = time.monotonic() + 30
    try:
        editor._start()
        editor.conversation._ensure_agent_ready()
        assert editor.conversation.agent.include_default_tools == ['FinishTool']
    finally:
        editor.close()


@pytest.mark.skipif(os.environ.get('MLSURGIE_DOCKER_TEST') != '1', reason='requires running Docker and local construction image')
def test_actual_container_tooling_persists_and_delivers(monkeypatch, tmp_path):
    monkeypatch.setenv('MLSURGIE_TEST_KEY', 'fixture-no-provider-call')
    def runtime(editor, context, timeout):
        if context['iteration'] == 1:
            # The agent writes its own reusable tool, then uses it from a separate execution.
            out = editor.execute("from pathlib import Path\nPath('/workspace/bench.py').write_text("
                                 "'import json, pandas as pd\\n"
                                 "def load():\\n    return pd.read_csv(\"/inputs/data/support.csv\")\\n')\n"
                                 "print('tool saved')")
            assert 'tool saved' in out
        source = '''import json
from bench import load
df = load()
task = json.load(open('/inputs/task.json'))
state = json.load(open('/inputs/state.json'))
p = {'base': 'zero', 'hidden_sizes': [], 'rationale': 'Sign fixture', 'operations': []}
if state['iteration'] == 2:
    pre = task['preprocessing'][0]
    p['hidden_sizes'] = [2]
    p['operations'] = [
        {'op':'set','tensor':'layers.0.weight','value':[[1,0],[-1,0]]},
        {'op':'set','tensor':'layers.1.weight','value':[[8*pre['scale'],-8*pre['scale']]]},
        {'op':'set','tensor':'layers.1.bias','value':[8*pre['mean']]},
    ]
print('rows', len(df))
json.dump(p, open('/workspace/candidate.json', 'w'))
'''
        assert 'captured' in editor.execute(source)
        assert 'ls-ok' in editor.execute('ls /skills >/dev/null && echo ls-ok', 'shell')
        net = editor.execute("import urllib.request\nurllib.request.urlopen('http://example.com', timeout=3)")
        assert 'failed' in net
        return {'input_tokens': 0, 'output_tokens': 0}
    editor = OpenHandsEditor('fixture/no-model', tmp_path / 'run', 'MLSURGIE_TEST_KEY', runtime=runtime)
    editor.identity['kind'] = 'scripted_fixture'
    editor.check()
    result = build(table=data(), editor=editor, output=tmp_path / 'run', budget=Budget(iterations=2),
                   dynamic_architecture=True, isolated_evaluation=True)
    assert result.report['selected_iteration'] == 2
    assert result.report['final_test']['accuracy'] == 1
    assert replay(result.path).fingerprint() == load(result.path).network.fingerprint()


def test_final_evaluator_failure_preserves_frozen_model_and_error_report(monkeypatch, tmp_path):
    import mlsurgie.evaluator as worker
    from mlsurgie import ScriptedEditor
    calls = []
    def evaluate(model, x, timeout):
        calls.append(len(x))
        if len(calls) == 2:
            raise RuntimeError('worker unavailable')
        return model.outputs(x)
    monkeypatch.setattr(worker, 'evaluate', evaluate)
    result = build(table=data(), editor=ScriptedEditor([candidate([])]),
                   output=tmp_path / 'run', budget=Budget(iterations=1), isolated_evaluation=True)
    assert result.report['status'] == 'final_evaluation_error'
    assert result.report['final_test'] is None
    assert 'worker unavailable' in result.report['final_evaluation_error']
    assert load(result.path).network.fingerprint() == replay(result.path).fingerprint()
    assert (result.path / 'selection.json').exists()


@pytest.mark.parametrize('token_limit, expected_calls, selected', [(100000, 4, 2), (445, 1, 1)])
def test_actual_sdk_tool_loop_with_mock_model_and_independent_evaluation(monkeypatch, tmp_path, token_limit, expected_calls, selected):
    monkeypatch.setenv('OH_PERSISTENCE_DIR', str(tmp_path / 'sdk-state'))
    monkeypatch.setenv('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
    monkeypatch.setenv('OPENHANDS_SUPPRESS_BANNER', '1')
    sdk = pytest.importorskip('openhands.sdk')
    from litellm.types.utils import ModelResponse
    from openhands.sdk.llm import LLMResponse, Message
    from openhands.sdk.llm.message import MessageToolCall
    from openhands.sdk.llm.utils.metrics import MetricsSnapshot
    monkeypatch.setenv('MLSURGIE_TEST_KEY', 'fake-key-not-for-network')
    calls = []
    def completion(self, messages, tools=None, **kwargs):
        calls.append(messages)
        if len(calls) % 2:
            name = next(tool.name for tool in tools if tool.name != 'finish')
            args = {'source': '# known-rule fixture', 'language': 'python'}
        else:
            name, args = 'finish', {'message': 'Ready for independent evaluation.'}
        self.metrics.accumulated_token_usage.prompt_tokens += 10
        self.metrics.accumulated_token_usage.completion_tokens += 5
        message = Message(role='assistant', tool_calls=[MessageToolCall(
            id=f'call-{len(calls)}', name=name, arguments=json.dumps(args), origin='completion')])
        return LLMResponse(message=message, metrics=MetricsSnapshot(), raw_response=ModelResponse())
    monkeypatch.setattr(sdk.LLM, 'completion', completion)
    monkeypatch.setattr(sdk.LLM, 'resolve_runtime_metadata', lambda self: None)
    monkeypatch.setattr(sdk.LLM, 'get_token_count', lambda self, *args, **kwargs: 100)
    editor = OpenHandsEditor('openrouter/test/model', tmp_path / 'run', 'MLSURGIE_TEST_KEY', sandbox=FixtureSandbox())
    editor.identity['kind'] = 'scripted_fixture'
    try:
        result = build(table=data(), editor=editor, output=tmp_path / 'run',
                       budget=Budget(iterations=2, max_editor_tokens=token_limit, max_output_tokens=200),
                       dynamic_architecture=True, isolated_evaluation=True)
    finally:
        editor.close()
    assert result.report['status'] == 'completed', result.report
    assert result.report['selected_iteration'] == selected
    assert result.report['cost']['editor_tokens'] == expected_calls * 15
    assert len(calls) == expected_calls
    if expected_calls == 1:
        assert result.report['stop_reason'] == 'editor_token_preflight'
        checks = result.report['history'][0]['usage']['request_preflight']
        assert checks[0]['allowed'] is True and checks[1]['allowed'] is False
        assert checks[1]['reserved_tokens'] == 438
        assert load(result.path).network.hidden_sizes == []
    assert list((result.path / 'agent-events').rglob('*.json'))


def test_agent_can_finish_early_without_new_candidate_and_usage_is_retained(monkeypatch, tmp_path):
    monkeypatch.setenv('MLSURGIE_TEST_KEY', 'test')
    def runtime(editor, context, timeout):
        if context['iteration'] == 1:
            editor.execute('# fixture')
        else:
            editor.stop_requested = True
        return {'input_tokens': 10, 'output_tokens': 5}
    editor = OpenHandsEditor('test/model', tmp_path / 'run', 'MLSURGIE_TEST_KEY', sandbox=FixtureSandbox(), runtime=runtime)
    result = build(table=data(), editor=editor, output=tmp_path / 'run', budget=Budget(iterations=8), dynamic_architecture=True)
    assert result.report['stop_reason'] == 'editor_finished'
    assert result.report['selected_iteration'] == 1
    assert result.report['history'][1]['status'] == 'editor_stopped'
    assert result.report['cost']['editor_tokens'] == 30
    assert replay(result.path).fingerprint() == result.model.network.fingerprint()


def test_skills_are_packaged_and_indexed():
    from mlsurgie.harness import SKILLS_DIR
    index = (SKILLS_DIR / 'README.md').read_text()
    for skill in sorted(p.name for p in SKILLS_DIR.glob('*.md') if p.name != 'README.md'):
        assert f'`{skill}`' in index, skill
    assert not list(Path(SKILLS_DIR).glob('*.py')), 'skills are knowledge, not prebuilt scripts'
