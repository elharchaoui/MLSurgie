"""The weight-space programming protocol.

A run is: split -> support-only preprocessing -> editor proposals (weight-program IR)
-> independent development evaluation -> declared feedback -> declared selection
-> freeze -> one sealed final-test evaluation -> export.

Every experimental variable that changes what a result means is a declared, recorded
component: the task, Budget, FeedbackPolicy, SelectionRule, and Track.
"""
import json
import platform
import subprocess
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from . import tasks
from .artifacts import BuildResult, Predictor, save_diff, save_model, save_report, write_json
from .contracts import METRICS, Budget, improves, reaches
from .data import Preprocessor, split_table
from .metrics import objective
from .network import apply_program, program_guide, rebase_numeric_program
from .version import VERSION


@dataclass(frozen=True)
class FeedbackPolicy:
    """What the editor learns from each development evaluation.

    error_rows: rows shown per evaluation (None: budget.diagnostic_examples; 0: metrics only).
    max_revealed_labels: cumulative cap on development labels revealed through error rows.
    """
    error_rows: int | None = None
    max_revealed_labels: int | None = None

    def __post_init__(self):
        for value in (self.error_rows, self.max_revealed_labels):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("Feedback limits must be nonnegative integers")


@dataclass(frozen=True)
class SelectionRule:
    """Replace the best candidate only when the metric improves by more than min_delta."""
    metric: str | None = None
    min_delta: float = 0.0

    def __post_init__(self):
        if not isinstance(self.min_delta, (int, float)) or self.min_delta < 0:
            raise ValueError("min_delta must be nonnegative")


@dataclass(frozen=True)
class Track:
    """Comparability class of a result.

    names: "semantic" (real feature names/descriptions) or "anonymized" (f01..., no descriptions).
    policy: "strict" (code flagged by the fitting audit is rejected) or "open" (flags recorded only).
    """
    names: str = "semantic"
    policy: str = "strict"

    def __post_init__(self):
        if self.names not in {"semantic", "anonymized"} or self.policy not in {"strict", "open"}:
            raise ValueError("Track names must be semantic/anonymized; policy strict/open")


LIMITATIONS = [
    "Adaptive development evaluation can overfit.",
    "Accuracy interval is Wilson 95%, assumes independent observations; other metrics have no interval here.",
    "Custom Python editor plugins are trusted; isolation is enforced through built-in JSON editor contexts, not an OS sandbox.",
    "Token limits stop subsequent work; one provider request can overshoot due to input token usage.",
    "Scripted fixtures demonstrate the protocol, not autonomous LLM performance.",
]


def build(*, table, editor, output, budget=None, seed=42, dynamic_architecture=False,
          isolated_evaluation=False, feedback=None, selection=None, track=None, initial_program=None, initial_preprocessing=None):
    """Construct a network by direct weight programs; no numerical weight fitting by the framework.

    The built-in editors see only JSON contexts (or prepared files, for the agent harness).
    Custom in-process editors are trusted extensions, not a security boundary.
    initial_program: optional starting network (e.g. a faulty model in a repair task),
    evaluated as iteration 0 and offered to the editor as the current best.
    """
    budget = budget or Budget()
    feedback = feedback or FeedbackPolicy()
    selection = selection or SelectionRule()
    track = track or Track()
    if initial_preprocessing is not None and [p["name"] for p in initial_preprocessing] != [f.name for f in table.features]:
        raise ValueError("Historical preprocessing features must match the task in order")
    if track.names == "anonymized":
        from .tracks import anonymize
        table = anonymize(table)
    goal = table.goal
    metric = selection.metric or goal.primary_metric
    if metric not in METRICS[goal.task]:
        raise ValueError(f"Selection metric {metric} is not available for {goal.task}")
    error_rows = budget.diagnostic_examples if feedback.error_rows is None else feedback.error_rows
    classes = table.validate()
    out_size = tasks.output_size(goal, classes)
    splits = split_table(table, seed)
    support_rows = [table.rows[int(i)] for i in splits["support"]]
    pre = Preprocessor.from_support(support_rows, table.features, budget.max_input_features)
    if initial_preprocessing is not None:
        if initial_program is None:
            raise ValueError("initial_preprocessing requires initial_program")
        initial_program = rebase_numeric_program(initial_program, initial_preprocessing, pre.state, budget, out_size)
    xs = pre.transform(support_rows)
    ys = table.labels(splits["support"])
    development_rows = [table.rows[int(i)] for i in splits["development"]]
    xd = pre.transform(development_rows)
    # Clustering reference labels never enter development metrics or feedback.
    yd = None if goal.task == "clustering" else table.labels(splits["development"])
    test_count = len(splits["test"])
    dev_count = len(development_rows)
    # Reserve one final test evaluation before spending any development budget.
    if dev_count + test_count > budget.max_evaluated_rows:
        raise ValueError("Row budget must cover at least one development pass and final test")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "iterations").mkdir()
    identity = deepcopy(editor.identity)
    protocol = {"feedback": asdict(feedback), "selection": {"metric": metric, "min_delta": selection.min_delta},
                "track": asdict(track), "initial_program": initial_program is not None}
    manifest = {
        "format_version": 2, "task": goal.task, "goal": goal.to_dict(), "budget": asdict(budget),
        "editor": identity, "features": [asdict(f) for f in table.features],
        "preprocessing": pre.state, "encoded_features": pre.names, "input_size": len(pre.names),
        "output_size": out_size, "classes": classes, "dataset_hash": table.fingerprint(), "split_seed": seed,
        "split_indices": {k: v.tolist() for k, v in splits.items()}, "group_column": table.group_column,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "torch": str(torch.__version__), "mlsurgie": VERSION},
        "protocol": "direct_weight_programming; support-only preprocessing; final test isolated from editor",
        "protocol_components": protocol,
        "dynamic_architecture": dynamic_architecture,
        "isolated_evaluation": isolated_evaluation,
        "initial_preprocessing": initial_preprocessing,
    }
    write_json(output / "manifest.json", manifest)
    # Editors never learn even the name of a clustering reference column.
    visible_goal = {**goal.to_dict(), "reference_column": "hidden" if goal.reference_column else None}
    if hasattr(editor, "prepare"):
        # Agents receive full support data as files instead of a giant prompt.
        editor.prepare({"goal": visible_goal, "task": goal.task, "classes": classes, "output_size": out_size,
                        "features": [asdict(f) for f in table.features], "preprocessing": pre.state,
                        "encoded_features": pre.names, "support_rows": support_rows, "support_encoded": xs,
                        "support_labels": None if goal.task == "clustering" else ys,
                        "target": goal.target, "limits": asdict(budget), "track": asdict(track),
                        "protocol": protocol})
    support_examples = []
    for j, row in enumerate(support_rows[:budget.support_examples]):
        example = {"features": {f.name: row[f.name] for f in table.features}, "encoded": xs[j].tolist()}
        if goal.task == "regression":
            example["target"] = float(ys[j])
        elif goal.task != "clustering":
            example["label"] = int(ys[j])
        support_examples.append(example)
    context_base = {
        "task": goal.task, "goal": visible_goal, "limits": asdict(budget),
        "program_format": program_guide(goal.task, dynamic_architecture),
        "encoded_features": pre.names, "preprocessing": pre.state, "classes": classes, "output_size": out_size,
        "support_count": len(support_rows), "support_examples": support_examples,
        "selection": {"metric": metric, "min_delta": selection.min_delta},
        "feedback_role": "development, used adaptively; final test is inaccessible",
        "track": asdict(track),
    }
    if goal.task == "binary":
        context_base["support_positive_fraction"] = float(ys.mean())
    if goal.task == "regression":
        context_base["support_target"] = {"mean": float(ys.mean()), "std": float(ys.std()),
                                          "min": float(ys.min()), "max": float(ys.max())}

    state = {"best": None, "metrics": None, "iteration": None, "value": None}
    frozen = None
    history = []
    last_feedback = None
    rows_used = 0
    tokens_used = 0
    revealed = 0
    stop_reason = "iteration_budget"
    started = time.monotonic()

    def run_inference(model, x, deadline):
        if isolated_evaluation:
            from . import evaluator
            return evaluator.evaluate(model, x, deadline)
        return model.outputs(x)

    def evaluate_candidate(candidate, folder, base, iteration):
        nonlocal rows_used, revealed, frozen
        rows_used += dev_count
        raw = run_inference(candidate, xd, max(.1, budget.max_seconds - (time.monotonic() - started)))
        metrics = tasks.evaluate(goal, yd, raw, xd)
        value = objective(metrics, metric)
        delta = save_diff(folder, candidate, base)
        frozen = candidate.hidden_sizes
        reference = None if state["best"] is None else state["metrics"][metric]
        selected = improves(metric, metrics[metric], reference, selection.min_delta)
        if selected:
            state.update(best=candidate, metrics=metrics, iteration=iteration, value=value)
        rows = []
        allowance = error_rows
        if tasks.reveals_labels(goal) and feedback.max_revealed_labels is not None:
            allowance = min(allowance, max(0, feedback.max_revealed_labels - revealed))
        for i, detail in tasks.worst_rows(goal, yd, raw, allowance):
            rows.append({"features": {f.name: development_rows[i][f.name] for f in table.features},
                         "encoded": xd[i].tolist(), **detail})
        if tasks.reveals_labels(goal):
            revealed += len(rows)
        result = {"status": "evaluated", "metrics": metrics, "selected": selected, "largest_errors": rows,
                  "revealed_development_labels": revealed if tasks.reveals_labels(goal) else 0}
        if allowance < error_rows:
            result["note"] = "Development label budget exhausted; metrics only."
        return metrics, selected, delta, result

    if initial_program is not None:
        folder = output / "iterations" / "000"
        folder.mkdir()
        write_json(folder / "proposal.json", initial_program)
        candidate = apply_program(initial_program, len(pre.names), budget, None, None, out_size)
        metrics, selected, delta, last_feedback = evaluate_candidate(candidate, folder, None, 0)
        last_feedback["status"] = "initial_model"
        record = {"iteration": 0, "status": "evaluated", "selected": True, "initial": True,
                  "state_hash": candidate.fingerprint(), "metrics": metrics, "diff": delta}
        history.append(record)
        write_json(folder / "result.json", record)

    for iteration in range(1, budget.iterations + 1):
        remaining = budget.max_seconds - (time.monotonic() - started)
        if remaining <= 0:
            stop_reason = "time_budget"
            break
        if tokens_used >= budget.max_editor_tokens:
            stop_reason = "editor_token_budget"
            break
        if rows_used + dev_count + test_count > budget.max_evaluated_rows:
            stop_reason = "evaluated_row_budget"
            break
        folder = output / "iterations" / f"{iteration:03d}"
        folder.mkdir()
        best = state["best"]
        context = {**deepcopy(context_base), "iteration": iteration,
                   "best": best.snapshot() if best else None,
                   "best_metrics": deepcopy(state["metrics"]),
                   "last_feedback": deepcopy(last_feedback),
                   "frozen_hidden_sizes": None if dynamic_architecture else frozen,
                   "output_token_limit": min(budget.max_output_tokens, budget.max_editor_tokens - tokens_used),
                   "remaining": {"iterations": budget.iterations - iteration + 1, "seconds": remaining,
                                 "editor_tokens": budget.max_editor_tokens - tokens_used}}
        write_json(folder / "context.json", context)
        record = {"iteration": iteration, "selected": False}
        try:
            reply = editor.propose(deepcopy(context), timeout=remaining)
        except StopIteration:
            stop_reason = "editor_exhausted"
            break
        except Exception as exc:
            record.update(status="editor_error", error=f"{type(exc).__name__}: {exc}")
            history.append(record)
            write_json(folder / "result.json", record)
            stop_reason = "editor_error"
            break
        usage = reply.usage
        try:
            used = sum(int(usage.get(k, 0)) for k in ("input_tokens", "output_tokens"))
            if used < 0:
                raise ValueError
        except (TypeError, ValueError):
            record.update(status="editor_error", error="Editor reported invalid token usage")
            history.append(record)
            write_json(folder / "result.json", record)
            stop_reason = "editor_error"
            break
        tokens_used += used
        record["usage"] = usage
        if reply.raw is not None:
            (folder / "reply.txt").write_text(reply.raw)
        if reply.audit is not None:
            record["audit"] = reply.audit
        if reply.stop and reply.program is None:
            record.update(status="editor_stopped")
            history.append(record)
            write_json(folder / "result.json", record)
            stop_reason = reply.stop_reason or "editor_finished"
            break
        try:
            serialized = json.dumps(reply.program, allow_nan=False)
            if len(serialized.encode()) > 2_000_000:
                raise ValueError("Program exceeds 2 MB")
            write_json(folder / "proposal.json", reply.program)
            if time.monotonic() - started > budget.max_seconds or tokens_used > budget.max_editor_tokens:
                record.update(status="budget_exceeded", error="Editor response exceeded runtime/token budget")
                history.append(record)
                write_json(folder / "result.json", record)
                stop_reason = "editor_budget_exceeded"
                break
            violations = (reply.audit or {}).get("violations") or []
            if violations and track.policy == "strict":
                record.update(status="policy_violation", error="Fitting audit flagged construction code")
                last_feedback = {"status": "policy_violation", "violations": violations,
                                 "message": "Strict track: rewrite the construction without the flagged fitting patterns."}
            else:
                candidate = apply_program(reply.program, len(pre.names), budget, best,
                                          None if dynamic_architecture else frozen, out_size)
                base = best if reply.program["base"] == "best" else None
                metrics, selected, delta, last_feedback = evaluate_candidate(candidate, folder, base, iteration)
                record.update(status="evaluated", selected=selected, state_hash=candidate.fingerprint(),
                              metrics=metrics, diff=delta)
        except (ValueError, TypeError, KeyError, RuntimeError, OverflowError, subprocess.TimeoutExpired) as exc:
            record.update(status="invalid", error=f"{type(exc).__name__}: {exc}")
            last_feedback = {"status": "invalid", "error": record["error"]}
        history.append(record)
        write_json(folder / "result.json", record)
        if reply.stop:
            stop_reason = reply.stop_reason or "editor_finished"
            break
        if state["metrics"] and goal.target_score is not None and reaches(metric, state["metrics"][metric], goal.target_score):
            stop_reason = "development_target_reached"
            break

    best = state["best"]
    final_metrics = None
    final_evaluation_error = None
    predictor = None
    if best is not None:
        # No more editor calls occur after this point. Freeze before opening test data.
        write_json(output / "selection.json", {"iteration": state["iteration"], "state_hash": best.fingerprint(),
                                               "primary_metric": metric})
        test_rows = [table.rows[int(i)] for i in splits["test"]]
        xt = pre.transform(test_rows)
        try:
            raw = run_inference(best, xt, max(30., budget.max_seconds))
            final_metrics = tasks.evaluate(goal, table.labels(splits["test"]), raw, xt)
            rows_used += len(test_rows)
        except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
            final_evaluation_error = f"{type(exc).__name__}: {exc}"
        save_model(output, best, pre, classes, goal.threshold, budget, goal.task)
        predictor = Predictor(best, pre, classes, goal.threshold, goal.task)
    target_met = None
    if final_metrics is not None and goal.target_score is not None:
        target_met = reaches(metric, final_metrics[metric], goal.target_score)
    report = {
        "status": ("no_valid_model" if best is None else "final_evaluation_error" if final_evaluation_error
                   else "target_not_met" if target_met is False else "completed"),
        "task": goal.task, "selection_metric": metric,
        "final_evaluation_error": final_evaluation_error,
        "editor": identity, "stop_reason": stop_reason, "selected_iteration": state["iteration"],
        "assurance": identity.get("assurance", "validated JSON operations; editor derivation is unverified"),
        "track": asdict(track),
        "development": state["metrics"], "final_test": final_metrics, "final_target_met": target_met,
        "parameters": None if best is None else sum(p.numel() for p in best.parameters()),
        "architecture": None if best is None else best.architecture(),
        "history": history, "cost": {"evaluated_rows": rows_used, "editor_tokens": tokens_used,
                                    "revealed_development_labels": revealed,
                                    "wall_seconds": time.monotonic() - started,
                                    "currency_cost": (round(sum(h.get("usage", {}).get("cost_usd", 0) for h in history), 6)
                                                      if any("cost_usd" in h.get("usage", {}) for h in history) else None)},
        "limitations": list(LIMITATIONS),
    }
    save_report(output, report)
    return BuildResult(output, report, predictor)
