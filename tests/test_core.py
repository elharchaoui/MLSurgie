import json
from copy import deepcopy

import numpy as np
import pytest

from mlsurgie import Budget, Feature, Goal, ScriptedEditor, Table, build, load, replay
from mlsurgie.data import Preprocessor, split_table
from mlsurgie.demo import ThresholdFixture, demo
from mlsurgie.editors import OllamaEditor
from mlsurgie.metrics import score
from mlsurgie.network import Network, apply_program


def table():
    return Table([{"x": float(i - 50), "label": str(int(i >= 50))} for i in range(100)],
                 [Feature("x")], Goal("Predict known sign", "label"))


def zero(hidden=None):
    return {"base": "zero", "hidden_sizes": [2] if hidden is None else hidden, "rationale": "Control", "operations": []}


def test_complete_run_selects_best_not_last_and_replays(tmp_path):
    result = demo(tmp_path / "run")
    assert result.report["selected_iteration"] == 2
    assert result.report["history"][2]["selected"] is False
    assert result.report["final_test"]["accuracy"] == 1.
    assert result.report["development"]["log_loss"] < .2
    restored = load(result.path)
    replayed = replay(result.path)
    rows = [{"x": -1}, {"x": 1}, {"x": None}]
    np.testing.assert_array_equal(restored.predict_proba(rows), result.model.predict_proba(rows))
    assert replayed.fingerprint() == restored.network.fingerprint()
    assert all(not p.requires_grad for p in restored.network.parameters())


def test_editor_never_sees_test_data_and_preprocessing_uses_support(tmp_path):
    data = table()
    split = split_table(data)
    held_out = {data.rows[int(i)]["x"] for i in split["test"]}
    class Spy(ThresholdFixture):
        contexts = []
        def propose(self, context, timeout):
            self.contexts.append(deepcopy(context))
            for row in context["support_examples"]:
                assert row["features"]["x"] not in held_out
            for row in (context["last_feedback"] or {}).get("largest_probability_errors", []):
                assert row["features"]["x"] not in held_out
            assert "test" not in context
            return super().propose(context, timeout)
    spy = Spy()
    result = build(table=data, editor=spy, output=tmp_path / "run", budget=Budget(iterations=3))
    manifest = json.loads((result.path / "manifest.json").read_text())
    expected = np.mean([data.rows[int(i)]["x"] for i in split["support"]])
    assert manifest["preprocessing"][0]["mean"] == expected
    assert len(spy.contexts) == 3


def test_invalid_programs_consume_iterations_without_evaluation(tmp_path):
    bad = zero()
    bad["operations"] = [{"op": "exec", "tensor": "layers.0.weight", "value": "train()"}]
    result = build(table=table(), editor=ScriptedEditor([bad, zero()]), output=tmp_path / "run", budget=Budget(iterations=1))
    assert result.model is None
    assert result.report["history"][0]["status"] == "invalid"
    assert result.report["cost"]["evaluated_rows"] == 0
    assert not (result.path / "weights.npz").exists()


@pytest.mark.parametrize("operation", [
    {"op": "set", "tensor": "layers.0.weight", "value": [[1]]},
    {"op": "fill", "tensor": "layers.0.bias", "value": float("nan")},
    {"op": "set_entries", "tensor": "layers.0.weight", "entries": [{"index": [-1, 0], "value": 1}]},
    {"op": "set_entries", "tensor": "layers.0.weight", "entries": [{"index": [0, 0], "value": 1e20}]},
])
def test_reject_bad_patches_without_mutating_base(operation):
    best = Network(2, [2])
    before = best.fingerprint()
    program = {**zero(), "base": "best", "base_hash": before, "operations": [operation]}
    with pytest.raises(ValueError):
        apply_program(program, 2, Budget(), best, [2])
    assert best.fingerprint() == before


def test_stale_base_and_architecture_limits():
    best = Network(2, [2])
    with pytest.raises(ValueError, match="stale"):
        apply_program({**zero(), "base": "best", "base_hash": "wrong"}, 2, Budget(), best)
    with pytest.raises(ValueError, match="frozen"):
        apply_program(zero([3]), 2, Budget(), best, [2])
    with pytest.raises(ValueError, match="Parameter"):
        apply_program(zero([64, 64]), 20, Budget(max_parameters=100))


def test_missing_unknown_and_constant_features():
    features = [Feature("n"), Feature("c", "categorical")]
    pre = Preprocessor.from_support([{"n": 4, "c": "a"}, {"n": "", "c": "a"}], features)
    transformed = pre.transform([{"n": None, "c": "new"}, {"n": 4, "c": ""}])
    np.testing.assert_array_equal(transformed, [[0, 1, 0, 0, 1], [0, 0, 0, 1, 0]])
    with pytest.raises(ValueError):
        pre.transform([{"n": "inf", "c": "a"}])


def test_metrics_ties_and_zero_predictions():
    result = score([0, 1, 0, 1], [.5, .5, .5, .5])
    assert result["roc_auc"] == .5
    assert result["log_loss"] == pytest.approx(np.log(2))
    assert result["confusion_matrix"] == [[0, 2], [0, 2]]
    assert score([0, 1], [0, 1])["roc_auc"] == 1
    assert score([0, 1], [1, 0])["roc_auc"] == 0
    assert score([0, 0], [.1, .2])["roc_auc"] is None


def test_group_splits_disjoint_and_tiny_data_rejected():
    data = table()
    for i, row in enumerate(data.rows):
        row["group"] = str(i % 10)
    data.group_column = "group"
    parts = split_table(data)
    groups = [{data.rows[int(i)]["group"] for i in p} for p in parts.values()]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    data = table()
    data.rows = data.rows[:3] + data.rows[-3:]
    with pytest.raises(ValueError, match="five"):
        split_table(data)


def test_row_budget_reserves_final_test(tmp_path):
    result = build(table=table(), editor=ScriptedEditor([zero()] * 3), output=tmp_path / "run",
                   budget=Budget(iterations=3, max_evaluated_rows=40))
    assert len(result.report["history"]) == 1
    assert result.report["cost"]["evaluated_rows"] == 40
    assert result.report["stop_reason"] == "evaluated_row_budget"


def test_tampered_checkpoint_and_replay_rejected(tmp_path):
    result = demo(tmp_path / "run")
    proposal_path = result.path / "iterations/002/proposal.json"
    p = json.loads(proposal_path.read_text())
    p["operations"][2]["value"] = [100]
    proposal_path.write_text(json.dumps(p))
    with pytest.raises(ValueError, match="Replay"):
        replay(result.path)
    with (result.path / "weights.npz").open("ab") as file:
        file.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        load(result.path)


def test_ollama_payload_and_usage(monkeypatch):
    import mlsurgie.editors as editors
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, size):
            return json.dumps({"message": {"content": json.dumps(zero())}, "prompt_eval_count": 11, "eval_count": 22}).encode()
    def send(request, timeout):
        body = json.loads(request.data)
        assert body["stream"] is False and body["format"] == "json"
        assert body["options"]["num_predict"] == 100
        assert timeout == 12
        return Response()
    monkeypatch.setattr(editors, "urlopen", send)
    reply = OllamaEditor("test").propose({"output_token_limit": 100}, timeout=12)
    assert reply.program == zero()
    assert reply.usage == {"input_tokens": 11, "output_tokens": 22}


def test_valid_patch_diff_replay_and_final_target_failure(tmp_path):
    data = table()
    data.goal = Goal("Predict sign", "label", target_score=0.)
    patch = {**zero(), "base": "best", "base_hash": "$best", "operations": [
        {"op": "set_entries", "tensor": "layers.0.weight", "entries": [
            {"index": [0, 0], "value": 1}, {"index": [1, 0], "value": -1}]},
        {"op": "set", "tensor": "layers.1.weight", "value": [[3, -3]]},
    ]}
    result = build(table=data, editor=ScriptedEditor([zero(), patch]), output=tmp_path / "run", budget=Budget(iterations=2))
    assert result.report["status"] == "target_not_met"
    assert result.report["history"][1]["diff"]["changed_values"] == 4
    assert replay(result.path).fingerprint() == result.model.network.fingerprint()
    with np.load(result.path / "iterations/002/diff.npz", allow_pickle=False) as diff:
        np.testing.assert_array_equal(diff["layers.1.weight.old"], [0, 0])
        np.testing.assert_array_equal(diff["layers.1.weight.new"], [3, -3])


def test_token_overshoot_is_recorded_without_scoring(tmp_path):
    from mlsurgie.editors import Reply
    class Expensive:
        identity = {"kind": "mock"}
        def propose(self, context, timeout):
            return Reply(zero(), {"input_tokens": 101, "output_tokens": 1})
    result = build(table=table(), editor=Expensive(), output=tmp_path / "run", budget=Budget(max_editor_tokens=100))
    assert result.model is None
    assert result.report["cost"]["editor_tokens"] == 102
    assert result.report["history"][0]["status"] == "budget_exceeded"
    assert result.report["cost"]["evaluated_rows"] == 0


def test_cli_build_predict_and_replay(tmp_path):
    import csv
    import os
    import subprocess
    import sys
    data = tmp_path / "table.csv"
    with data.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "label"])
        writer.writeheader()
        writer.writerows(table().rows)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"goal": {"description": "Predict sign", "target": "label"},
                                  "features": [{"name": "x"}], "budget": {"iterations": 1}}))
    programs = tmp_path / "programs.json"
    programs.write_text(json.dumps([zero()]))
    env = {**os.environ, "PYTHONPATH": "src"}
    def run(*args):
        return subprocess.run([sys.executable, "-m", "mlsurgie.cli", *map(str, args)], env=env, text=True, capture_output=True, check=True)
    destination = tmp_path / "run"
    run("build", "--config", config, "--dataset", data, "--programs", programs, "--output", destination)
    run("replay", destination)
    output = tmp_path / "predictions.csv"
    run("predict", destination, "--dataset", data, "--output", output)
    with output.open() as handle:
        assert len(list(csv.DictReader(handle))) == 100
