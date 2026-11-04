"""Harness protocol tests use explicit fixtures, never paid model calls."""
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pytest

from mlsurgie import Budget, Feature, Goal, Table, build, load, replay
from mlsurgie.harness import OpenHandsEditor, program_guide
from mlsurgie.sandbox import DockerSandbox


def data():
    return Table([{'x': i - 50, 'label': str(int(i >= 50))} for i in range(100)],
                 [Feature('x')], Goal('Predict sign by direct weights', 'label'))


def candidate(width):
    return {'base': 'zero', 'hidden_sizes': width,
            'rationale': 'Known-rule protocol fixture', 'operations': []}


class FixtureSandbox:
    def check(self): pass

    def execute(self, source, context, directory, timeout):
        directory.mkdir(parents=True)
        (directory / 'execution.log').write_text('Fixture, not code execution')
        (directory / 'source.py').write_text(source)
        program = candidate([] if context['iteration'] == 1 else [2] if context['iteration'] == 2 else [3])
        if context['iteration'] == 2:
            pre = context['preprocessing'][0]
            program['operations'] = [
                {'op': 'set', 'tensor': 'layers.0.weight', 'value': [[1, 0], [-1, 0]]},
                {'op': 'set', 'tensor': 'layers.1.weight', 'value': [[8 * pre['scale'], -8 * pre['scale']]]},
                {'op': 'set', 'tensor': 'layers.1.bias', 'value': [8 * pre['mean']]},
            ]
        return program


def test_harness_revises_topology_independent_evaluation_and_replay(monkeypatch, tmp_path):
    monkeypatch.setenv('MLSURGIE_TEST_KEY', 'never-write-this-secret')
    contexts = []
    def runtime(editor, context, timeout):
        contexts.append(context)
        assert 'frozen' not in context['program_format'].lower()
        assert context['frozen_hidden_sizes'] is None
        assert 'test' not in context
        assert 'captured' in editor.execute('# explicit test fixture')
        return {'input_tokens': 10, 'output_tokens': 20}
    editor = OpenHandsEditor('test/fixture', tmp_path / 'run', 'MLSURGIE_TEST_KEY',
                             sandbox=FixtureSandbox(), runtime=runtime)
    editor.identity['kind'] = 'scripted_fixture'
    result = build(table=data(), editor=editor, output=tmp_path / 'run',
                   budget=Budget(iterations=3), dynamic_architecture=True, isolated_evaluation=True)
    assert result.report['selected_iteration'] == 2
    assert result.report['history'][2]['status'] == 'evaluated'
    assert result.report['history'][2]['selected'] is False
    assert result.report['final_test']['accuracy'] == 1
    assert result.report['cost']['editor_tokens'] == 90
    assert len(contexts) == 3
    predictor = load(result.path)
    assert predictor.network.hidden_sizes == [2]
    assert replay(result.path).fingerprint() == predictor.network.fingerprint()
    np.testing.assert_array_equal(predictor.predict([{'x': -1}, {'x': 1}]), ['0', '1'])
    for path in result.path.rglob('*.json'):
        assert 'never-write-this-secret' not in path.read_text()


def test_harness_failed_code_does_not_reuse_old_candidate(monkeypatch, tmp_path):
    monkeypatch.setenv('MLSURGIE_TEST_KEY', 'test')
    class Failing(FixtureSandbox):
        def execute(self, source, context, directory, timeout):
            raise ValueError('failure')
    def runtime(editor, context, timeout):
        editor.pending = candidate([])
        assert 'failed' in editor.execute('bad code')
        return {'input_tokens': 1, 'output_tokens': 1}
    editor = OpenHandsEditor('test/model', tmp_path / 'run', 'MLSURGIE_TEST_KEY',
                             sandbox=Failing(), runtime=runtime)
    result = build(table=data(), editor=editor, output=tmp_path / 'run', budget=Budget(iterations=1),
                   dynamic_architecture=True)
    assert result.model is None
    assert result.report['history'][0]['status'] == 'invalid'


def test_container_receives_no_network_keys_test_data_or_host_root(monkeypatch, tmp_path):
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        if command[1] == 'run':
            (tmp_path / 'work' / 'candidate.json').write_text(json.dumps(candidate([])))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr('mlsurgie.sandbox.subprocess.run', run)
    context = {'support_examples': [], 'goal': {'description': 'fixture'}}
    assert DockerSandbox().execute('pass', context, tmp_path, 12)['hidden_sizes'] == []
    cmd = commands[0]
    assert cmd[cmd.index('--network') + 1] == 'none'
    assert '--read-only' in cmd and '--cap-drop' in cmd and '--pids-limit' in cmd
    assert '--memory' in cmd and '--cpus' in cmd
    assert '--env' not in cmd and '-e' not in cmd and '--privileged' not in cmd
    assert 'docker.sock' not in ' '.join(cmd)
    assert (tmp_path / 'inputs' / 'task.json').read_text() == json.dumps(context)
    assert commands[-1][1:3] == ['rm', '-f']


def test_container_timeout_cleans_up(monkeypatch, tmp_path):
    commands = []
    def run(cmd, **kwargs):
        commands.append(cmd)
        if cmd[1] == 'run': raise subprocess.TimeoutExpired(cmd, 1)
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr('mlsurgie.sandbox.subprocess.run', run)
    with pytest.raises(ValueError, match='deadline'):
        DockerSandbox().execute('pass', {}, tmp_path, 1)
    assert commands[-1][1:3] == ['rm', '-f']


def test_container_rejects_symlink_artifact(monkeypatch, tmp_path):
    def run(cmd, **kwargs):
        if cmd[1] == 'run':
            (tmp_path / 'work' / 'candidate.json').symlink_to(tmp_path / 'inputs' / 'task.json')
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr('mlsurgie.sandbox.subprocess.run', run)
    with pytest.raises(ValueError, match='regular'):
        DockerSandbox().execute('pass', {}, tmp_path, 1)


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
def test_actual_container_construction_and_delivery(monkeypatch, tmp_path):
    monkeypatch.setenv('MLSURGIE_TEST_KEY', 'fixture-no-provider-call')
    def runtime(editor, context, timeout):
        source = '''import json
from pathlib import Path
ctx = json.loads(Path('/inputs/task.json').read_text())
p = {'base': 'zero', 'hidden_sizes': [], 'rationale': 'Known sign fixture', 'operations': []}
if ctx['iteration'] == 2:
    pre = ctx['preprocessing'][0]
    p['hidden_sizes'] = [2]
    p['operations'] = [
        {'op':'set','tensor':'layers.0.weight','value':[[1,0],[-1,0]]},
        {'op':'set','tensor':'layers.1.weight','value':[[8*pre['scale'],-8*pre['scale']]]},
        {'op':'set','tensor':'layers.1.bias','value':[8*pre['mean']]},
    ]
Path('/work/candidate.json').write_text(json.dumps(p))
'''
        assert 'captured' in editor.execute(source)
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
    from mlsurgie import ScriptedEditor
    import mlsurgie.evaluator as worker
    calls = []
    def evaluate(model, x, timeout):
        calls.append(len(x))
        if len(calls) == 2:
            raise RuntimeError('worker unavailable')
        return model.probabilities(x)
    monkeypatch.setattr(worker, 'evaluate', evaluate)
    result = build(table=data(), editor=ScriptedEditor([candidate([])]),
                   output=tmp_path / 'run', budget=Budget(iterations=1), isolated_evaluation=True)
    assert result.report['status'] == 'final_evaluation_error'
    assert result.report['final_test'] is None
    assert 'worker unavailable' in result.report['final_evaluation_error']
    assert load(result.path).network.fingerprint() == replay(result.path).fingerprint()
    assert (result.path / 'selection.json').exists()


def test_actual_sdk_tool_loop_with_mock_model_and_independent_evaluation(monkeypatch, tmp_path):
    monkeypatch.setenv('OH_PERSISTENCE_DIR', str(tmp_path / 'sdk-state'))
    monkeypatch.setenv('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
    monkeypatch.setenv('OPENHANDS_SUPPRESS_BANNER', '1')
    sdk = pytest.importorskip('openhands.sdk')
    from openhands.sdk.llm import LLMResponse, Message
    from openhands.sdk.llm.message import MessageToolCall
    from openhands.sdk.llm.utils.metrics import MetricsSnapshot
    from litellm.types.utils import ModelResponse
    monkeypatch.setenv('MLSURGIE_TEST_KEY', 'fake-key-not-for-network')
    calls = []
    def completion(self, messages, tools=None, **kwargs):
        calls.append(messages)
        if len(calls) % 2:
            name = next(tool.name for tool in tools if tool.name != 'finish')
            args = {'source': '# known-rule fixture'}
        else:
            name, args = 'finish', {'message': 'Ready for independent evaluation.'}
        self.metrics.accumulated_token_usage.prompt_tokens += 10
        self.metrics.accumulated_token_usage.completion_tokens += 5
        message = Message(role='assistant', tool_calls=[MessageToolCall(
            id=f'call-{len(calls)}', name=name, arguments=json.dumps(args), origin='completion')])
        return LLMResponse(message=message, metrics=MetricsSnapshot(), raw_response=ModelResponse())
    monkeypatch.setattr(sdk.LLM, 'completion', completion)
    monkeypatch.setattr(sdk.LLM, 'resolve_runtime_metadata', lambda self: None)
    editor = OpenHandsEditor('openrouter/test/model', tmp_path / 'run', 'MLSURGIE_TEST_KEY',
                             sandbox=FixtureSandbox())
    editor.identity['kind'] = 'scripted_fixture'
    try:
        result = build(table=data(), editor=editor, output=tmp_path / 'run',
                       budget=Budget(iterations=2), dynamic_architecture=True, isolated_evaluation=True)
    finally:
        editor.close()
    assert result.report['status'] == 'completed', result.report
    assert result.report['selected_iteration'] == 2
    assert result.report['cost']['editor_tokens'] == 60
    assert len(calls) == 4
    assert list((result.path / 'agent-events').rglob('*.json'))


def test_agent_can_finish_early_without_new_candidate_and_usage_is_retained(monkeypatch, tmp_path):
    monkeypatch.setenv('MLSURGIE_TEST_KEY', 'test')
    def runtime(editor, context, timeout):
        if context['iteration'] == 1:
            editor.execute('# fixture')
        else:
            editor.stop_requested = True
        return {'input_tokens': 10, 'output_tokens': 5}
    editor = OpenHandsEditor('test/model', tmp_path / 'run', 'MLSURGIE_TEST_KEY',
                             sandbox=FixtureSandbox(), runtime=runtime)
    result = build(table=data(), editor=editor, output=tmp_path / 'run',
                   budget=Budget(iterations=8), dynamic_architecture=True)
    assert result.report['stop_reason'] == 'editor_finished'
    assert result.report['selected_iteration'] == 1
    assert result.report['history'][1]['status'] == 'editor_stopped'
    assert result.report['cost']['editor_tokens'] == 30
    assert replay(result.path).fingerprint() == result.model.network.fingerprint()
