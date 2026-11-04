"""OpenRouter protocol tests; never make paid or external requests."""
import json
from urllib.error import HTTPError

import pytest

import mlsurgie.editors as editors
from mlsurgie import Budget, Feature, Goal, OpenRouterEditor, Table, build, replay

PROGRAM = {"base": "zero", "hidden_sizes": [], "rationale": "Control", "operations": []}


def mock_reply(monkeypatch, content=None, usage=None, **extra):
    body = {"choices": [{"message": {"content": json.dumps(PROGRAM) if content is None else content}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8} if usage is None else usage,
            "model": "test/resolved", **extra}
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, size): return json.dumps(body).encode()
    requests = []
    def send(request, timeout):
        requests.append((request, timeout))
        return Response()
    monkeypatch.setattr(editors, "urlopen", send)
    return requests


def test_missing_key_fails_before_network(monkeypatch):
    monkeypatch.delenv("MLSURGIE_TEST_KEY", raising=False)
    with pytest.raises(ValueError, match="MLSURGIE_TEST_KEY"):
        OpenRouterEditor("vendor/model", "MLSURGIE_TEST_KEY")


def test_authenticated_payload_usage_and_no_key_in_identity(monkeypatch):
    monkeypatch.setenv("MLSURGIE_TEST_KEY", "test-secret-value")
    requests = mock_reply(monkeypatch)
    editor = OpenRouterEditor("vendor/model", "MLSURGIE_TEST_KEY")
    reply = editor.propose({"output_token_limit": 128, "support_examples": []}, 9)
    request, timeout = requests[0]
    payload = json.loads(request.data)
    assert request.full_url == "https://openrouter.ai/api/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer test-secret-value"
    assert payload["model"] == "vendor/model"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["provider"]["require_parameters"] is True
    assert payload["max_tokens"] == 128 and payload["stream"] is False
    assert "tools" not in payload and timeout == 9
    assert reply.program == PROGRAM
    assert reply.usage == {"input_tokens": 12, "output_tokens": 8, "response_model": "test/resolved"}
    assert "test-secret-value" not in json.dumps(editor.identity)


def test_malformed_program_preserves_usage(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    mock_reply(monkeypatch, content="not valid JSON")
    reply = OpenRouterEditor("vendor/model").propose({"output_token_limit": 128}, 9)
    assert reply.program == {"invalid_json": "not valid JSON"}
    assert reply.usage["output_tokens"] == 8


@pytest.mark.parametrize("response", [
    {"error": {"message": "private provider details"}},
    {"choices": []},
    {"choices": [{"message": {"content": None}}]},
    {"usage": {"prompt_tokens": -1}},
])
def test_provider_errors_are_sanitized(monkeypatch, response):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    mock_reply(monkeypatch, **response)
    with pytest.raises(RuntimeError) as error:
        OpenRouterEditor("vendor/model").propose({"output_token_limit": 128}, 9)
    assert "private provider details" not in str(error.value)


def test_http_error_is_sanitized_without_retry(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls = []
    def fail(request, timeout):
        calls.append(request)
        raise HTTPError(request.full_url, 401, "private provider details", {}, None)
    monkeypatch.setattr(editors, "urlopen", fail)
    with pytest.raises(RuntimeError, match=r"HTTP 401") as error:
        OpenRouterEditor("vendor/model").propose({"output_token_limit": 128}, 9)
    assert "private" not in str(error.value) and len(calls) == 1


def test_openrouter_run_records_identity_usage_and_replays(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret-value")
    mock_reply(monkeypatch)
    table = Table([{"x": i, "label": str(i % 2)} for i in range(50)],
                  [Feature("x")], Goal("Predict parity", "label"))
    result = build(table=table, editor=OpenRouterEditor("vendor/model"),
                   output=tmp_path / "run", budget=Budget(iterations=1))
    assert result.report["editor"]["kind"] == "openrouter"
    assert result.report["cost"]["editor_tokens"] == 20
    assert result.report["final_test"]["n"] == 10
    assert replay(result.path).fingerprint() == result.model.network.fingerprint()
    for path in result.path.rglob('*'):
        if path.suffix in {'.json', '.txt', '.html'}:
            assert 'test-secret-value' not in path.read_text()


def test_cli_openrouter_selection(monkeypatch, tmp_path, capsys):
    from mlsurgie.cli import main
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    requests = mock_reply(monkeypatch)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"goal": {"description": "Predict parity", "target": "label"},
                                  "features": [{"name": "x"}], "budget": {"iterations": 1}}))
    dataset = tmp_path / "data.csv"
    dataset.write_text('x,label\n' + ''.join(f'{i},{i % 2}\n' for i in range(50)))
    args = ['mlsurgie', 'build', '--config', str(config), '--dataset', str(dataset),
            '--output', str(tmp_path / 'run'), '--openrouter-model', 'vendor/model']
    monkeypatch.setattr('sys.argv', args)
    main()
    assert len(requests) == 1
    assert json.loads(capsys.readouterr().out)['status'] == 'completed'
    monkeypatch.setattr('sys.argv', args + ['--ollama-model', 'local'])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
