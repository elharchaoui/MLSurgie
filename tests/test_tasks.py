"""Multi-task protocol, declared protocol components, tracks, baselines, and suites."""
import json
from pathlib import Path

import numpy as np
import pytest

from mlsurgie import (
    Budget,
    Feature,
    FeedbackPolicy,
    Goal,
    RandomSearchEditor,
    ScriptedEditor,
    SelectionRule,
    Table,
    build,
    load,
    replay,
)
from mlsurgie.metrics import adjusted_rand, average_precision_score, score_regression, silhouette
from mlsurgie.tracks import anonymize

ROOT = Path(__file__).resolve().parents[1]


def blobs(n=120, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [6, 0], [0, 6]])
    k = rng.integers(0, 3, n)
    x = centers[k] + rng.normal(0, .5, (n, 2))
    return [{"a": float(p[0]), "b": float(p[1]), "kind": f"k{c}"} for p, c in zip(x, k)]


def program(hidden, operations, rationale="Fixture mechanism"):
    return {"base": "zero", "hidden_sizes": hidden, "rationale": rationale, "operations": operations}


def test_multiclass_prototype_network_end_to_end(tmp_path):
    rows = blobs()
    table = Table(rows, [Feature("a"), Feature("b")], Goal("Which blob", "kind", task="multiclass"))
    class Prototype:
        identity = {"kind": "scripted_fixture"}
        def propose(self, context, timeout):
            assert context["output_size"] == 3 and context["classes"] == ["k0", "k1", "k2"]
            pre = {p["name"]: p for p in context["preprocessing"]}
            centers = np.array([[0, 0], [6, 0], [0, 6]], dtype=float)
            z = (centers - [pre["a"]["mean"], pre["b"]["mean"]]) / [pre["a"]["scale"], pre["b"]["scale"]]
            weight = [[4 * c[0], 0, 4 * c[1], 0] for c in z]
            bias = [-2 * float(c @ c) for c in z]
            return __import__("mlsurgie").editors.Reply(program([], [
                {"op": "set", "tensor": "layers.0.weight", "value": weight},
                {"op": "set", "tensor": "layers.0.bias", "value": bias}]), {"input_tokens": 0, "output_tokens": 0}, stop=True)
    result = build(table=table, editor=Prototype(), output=tmp_path / "run", budget=Budget(iterations=1))
    assert result.report["final_test"]["accuracy"] == 1.0
    predictor = load(result.path)
    assert list(predictor.predict([{"a": 6, "b": 0}, {"a": 0, "b": 6}])) == ["k1", "k2"]
    assert predictor.predict_proba([{"a": 0, "b": 0}]).shape == (1, 3)
    assert replay(result.path).fingerprint() == predictor.network.fingerprint()


def test_regression_linear_network_and_tanh_activation(tmp_path):
    rows = [{"x": float(i), "y": 3.0 * i + 5} for i in range(60)]
    table = Table(rows, [Feature("x")], Goal("Linear response", "y", task="regression"))
    class Linear:
        identity = {"kind": "scripted_fixture"}
        calls = 0
        def propose(self, context, timeout):
            self.calls += 1
            pre = context["preprocessing"][0]
            if self.calls == 1:
                return __import__("mlsurgie").editors.Reply(program([2], [], "tanh control") | {"activations": ["tanh"]},
                                                            {"input_tokens": 0, "output_tokens": 0})
            return __import__("mlsurgie").editors.Reply(program([], [
                {"op": "set", "tensor": "layers.0.weight", "value": [[3.0 * pre["scale"], 0]]},
                {"op": "set", "tensor": "layers.0.bias", "value": [3.0 * pre["mean"] + 5]}]),
                {"input_tokens": 0, "output_tokens": 0})
    result = build(table=table, editor=Linear(), output=tmp_path / "run", budget=Budget(iterations=2), dynamic_architecture=True)
    assert result.report["history"][0]["status"] == "evaluated"
    assert result.report["selected_iteration"] == 2
    assert result.report["final_test"]["rmse"] < 1e-3
    assert load(result.path).predict([{"x": 10}])[0] == pytest.approx(35, abs=1e-3)


def test_clustering_hides_reference_labels_until_final_report(tmp_path):
    rows = blobs()
    table = Table(rows, [Feature("a"), Feature("b")],
                  Goal("Three groups", task="clustering", n_clusters=3, reference_column="kind"))
    contexts = []
    class Centroids:
        identity = {"kind": "scripted_fixture"}
        def propose(self, context, timeout):
            contexts.append(context)
            pre = {p["name"]: p for p in context["preprocessing"]}
            centers = np.array([[0, 0], [6, 0], [0, 6]], dtype=float)
            z = (centers - [pre["a"]["mean"], pre["b"]["mean"]]) / [pre["a"]["scale"], pre["b"]["scale"]]
            return __import__("mlsurgie").editors.Reply(program([], [
                {"op": "set", "tensor": "layers.0.weight", "value": [[2 * c[0], 0, 2 * c[1], 0] for c in z]},
                {"op": "set", "tensor": "layers.0.bias", "value": [-float(c @ c) for c in z]}]),
                {"input_tokens": 0, "output_tokens": 0})
    result = build(table=table, editor=Centroids(), output=tmp_path / "run", budget=Budget(iterations=2))
    text = json.dumps(contexts)
    assert not any(f'"{v}"' in text for v in ("k0", "k1", "k2"))
    assert contexts[0]["goal"]["reference_column"] == "hidden"
    assert all("label" not in e for e in contexts[0]["support_examples"])
    assert result.report["development"]["adjusted_rand"] is None
    assert result.report["final_test"]["adjusted_rand"] == 1.0
    assert result.report["final_test"]["silhouette"] > .7


def test_feedback_policy_caps_revealed_labels_and_metrics_only(tmp_path):
    rows = [{"x": float(i), "label": str(i % 2)} for i in range(100)]
    table = Table(rows, [Feature("x")], Goal("Parity", "label"))
    seen = []
    class Spy(ScriptedEditor):
        def propose(self, context, timeout):
            seen.append(context["last_feedback"])
            return super().propose(context, timeout)
    build(table=table, editor=Spy([program([], [])] * 3), output=tmp_path / "a",
          budget=Budget(iterations=3, diagnostic_examples=8), feedback=FeedbackPolicy(max_revealed_labels=10))
    assert [len(f["largest_errors"]) for f in seen[1:]] == [8, 2]
    seen.clear()
    result = build(table=table, editor=Spy([program([], [])] * 2), output=tmp_path / "b",
                   budget=Budget(iterations=2), feedback=FeedbackPolicy(error_rows=0))
    assert seen[1]["largest_errors"] == [] and result.report["cost"]["revealed_development_labels"] == 0


def test_selection_rule_min_delta_and_metric(tmp_path):
    rows = [{"x": float(i - 50), "label": str(int(i >= 50))} for i in range(100)]
    table = Table(rows, [Feature("x")], Goal("Sign", "label"))
    weak = program([], [{"op": "set", "tensor": "layers.0.weight", "value": [[0.01, 0]]}])
    strong = program([], [{"op": "set", "tensor": "layers.0.weight", "value": [[0.02, 0]]}])
    result = build(table=table, editor=ScriptedEditor([weak, strong]), output=tmp_path / "r",
                   budget=Budget(iterations=2), selection=SelectionRule("roc_auc", min_delta=.01))
    assert result.report["selection_metric"] == "roc_auc"
    assert result.report["selected_iteration"] == 1  # equal AUC does not clear min_delta
    with pytest.raises(ValueError, match="not available"):
        build(table=table, editor=ScriptedEditor([weak]), output=tmp_path / "x", selection=SelectionRule("rmse"))


def test_repair_initial_program_is_iteration_zero_and_replays(tmp_path):
    rows = [{"x": float(i - 50), "label": str(int(i >= 50))} for i in range(100)]
    table = Table(rows, [Feature("x")], Goal("Sign", "label"))
    faulty = program([], [{"op": "set", "tensor": "layers.0.weight", "value": [[-1.0, 0]]}], "Deployed model with sign fault")
    patch = {**program([], [{"op": "set_entries", "tensor": "layers.0.weight", "entries": [{"index": [0, 0], "value": 1.0}]}]),
             "base": "best", "base_hash": "$best"}
    result = build(table=table, editor=ScriptedEditor([patch]), output=tmp_path / "r",
                   budget=Budget(iterations=1), initial_program=faulty)
    history = result.report["history"]
    assert history[0]["iteration"] == 0 and history[0]["initial"] is True
    assert result.report["selected_iteration"] == 1 and history[1]["diff"]["changed_values"] == 1
    assert result.report["final_test"]["accuracy"] == 1.0
    assert replay(result.path).fingerprint() == result.model.network.fingerprint()


def test_random_search_control_runs_every_task(tmp_path):
    rows = blobs()
    for i, (goal, features) in enumerate([
        (Goal("blob", "kind", task="multiclass"), [Feature("a"), Feature("b")]),
        (Goal("a from b", "a", task="regression"), [Feature("b")]),
        (Goal("groups", task="clustering", n_clusters=3), [Feature("a"), Feature("b")]),
    ]):
        result = build(table=Table(rows, features, goal), editor=RandomSearchEditor(seed=i), output=tmp_path / str(i),
                       budget=Budget(iterations=4))
        assert result.report["status"] == "completed"
        assert result.report["editor"]["kind"] == "search_baseline"
        assert replay(result.path).fingerprint() == result.model.network.fingerprint()


def test_anonymized_track_hides_names_and_values():
    table = Table([{"amount": 5, "merchant": "gift_cards", "fraud": "yes"}, {"amount": 1, "merchant": "grocery", "fraud": "no"}],
                  [Feature("amount", description="USD"), Feature("merchant", "categorical", "Merchant type")],
                  Goal("Detect fraud", "fraud", positive_label="yes"))
    anonymous = anonymize(table)
    text = json.dumps([anonymous.rows, [f.__dict__ for f in anonymous.features], anonymous.goal.to_dict()])
    for secret in ("amount", "merchant", "gift_cards", "fraud", "USD", "yes"):
        assert secret not in text
    assert anonymous.validate() == ["c1", "c2"] and anonymous.goal.positive_label == "c2"


def test_metrics_match_reference_implementations():
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    y, p = rng.integers(0, 2, 300), np.round(rng.random(300), 2)
    assert average_precision_score(y, p) == pytest.approx(sklearn.average_precision_score(y, p))
    a, b = rng.integers(0, 3, 200), rng.integers(0, 4, 200)
    assert adjusted_rand(a, b) == pytest.approx(sklearn.adjusted_rand_score(a, b))
    x = rng.normal(size=(150, 3))
    assert silhouette(x, b[:150]) == pytest.approx(sklearn.silhouette_score(x, b[:150]))
    t, f = rng.normal(size=50), rng.normal(size=50)
    assert score_regression(t, f)["r2"] == pytest.approx(sklearn.r2_score(t, f))


def test_missing_usage_is_refused(monkeypatch):
    import mlsurgie.editors as editors
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    body = {"choices": [{"message": {"content": "{}"}}], "model": "m"}
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, size): return json.dumps(body).encode()
    monkeypatch.setattr(editors, "urlopen", lambda request, timeout: Response())
    with pytest.raises(RuntimeError, match="token usage"):
        editors.OpenRouterEditor("vendor/model").propose({"output_token_limit": 10}, 5)
    body.update(message={"content": "{}"})
    with pytest.raises(RuntimeError, match="token usage"):
        editors.OllamaEditor("m").propose({"output_token_limit": 10}, 5)


def test_benchmark_tasks_load_and_baselines_run(tmp_path):
    pytest.importorskip("sklearn")
    from mlsurgie.config import load_task
    from mlsurgie.suite import run_suite
    for path in sorted((ROOT / "benchmarks" / "tasks").glob("*.json")):
        if path.name.endswith((".program.json", ".preprocessing.json")):
            continue
        table, options, config = load_task(path)
        table.validate()
        assert config["name"] == path.stem
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"name": "mini", "seeds": [0],
                                 "tasks": [str(ROOT / "benchmarks/tasks/wine.json"), str(ROOT / "benchmarks/tasks/segments_repair.json")]}))
    summary = run_suite(suite, lambda task, seed, out: RandomSearchEditor(seed), tmp_path / "out",
                        editor_options={"budget_overrides": {"iterations": 2}})
    wine, repair = summary["table"]
    assert wine["methods"]["logistic_regression"]["metrics"]["log_loss"]["mean"] < .5
    assert set(wine["methods"]) == {"editor", "prior", "logistic_regression", "random_forest", "svm_rbf"}
    assert repair["headline"] == "adjusted_rand" and "kmeans" in repair["methods"]
    assert "| **search_baseline** (editor) |" in (tmp_path / "out" / "results.md").read_text()


def test_published_v1_artifacts_still_verify():
    run = ROOT / "experiments" / "titanic" / "opus55"
    if not (run / "iterations").exists():
        pytest.skip("published run directory has no iteration records")
    assert replay(run).fingerprint() == json.loads((run / "selection.json").read_text())["state_hash"]


def test_spend_limits_are_opt_in_and_unlimited_by_default(tmp_path):
    budget = Budget()
    assert budget.max_editor_tokens is None and budget.max_seconds is None
    rows = [{"x": float(i - 50), "label": str(int(i >= 50))} for i in range(100)]
    seen = []
    class Costly(ScriptedEditor):
        def propose(self, context, timeout):
            seen.append(context["remaining"])
            reply = super().propose(context, timeout)
            reply.usage = {"input_tokens": 5_000_000, "output_tokens": 1_000_000}
            return reply
    result = build(table=Table(rows, [Feature("x")], Goal("Sign", "label")),
                   editor=Costly([program([], [])] * 3), output=tmp_path / "r", budget=Budget(iterations=3))
    assert [h["status"] for h in result.report["history"]] == ["evaluated"] * 3
    assert result.report["cost"]["editor_tokens"] == 18_000_000
    assert seen[0]["editor_tokens"] == "unlimited" and seen[0]["seconds"] == "unlimited"
    for task in sorted((ROOT / "benchmarks" / "tasks").glob("*.json")):
        config = json.loads(task.read_text())
        if isinstance(config, dict) and "budget" in config:
            assert not {"max_editor_tokens", "max_seconds"} & set(config["budget"]), task.name
