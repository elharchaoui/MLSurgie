from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .artifacts import BuildResult, Predictor, save_diff, save_model, save_report, write_json
from .contracts import Budget
from .data import Preprocessor, split_table
from .metrics import objective, score
from .network import PROGRAM_GUIDE, apply_program


def build(*, table, editor, output, budget=None, seed=42):
    """Construct a binary MLP by direct proposals; no numerical weight fitting.

    The built-in editor sees only JSON contexts. Custom in-process editors are
    trusted extensions, not a security boundary against malicious Python plugins.
    """
    budget = budget or Budget()
    classes = table.validate()
    splits = split_table(table, seed)
    support_rows = [table.rows[int(i)] for i in splits["support"]]
    pre = Preprocessor.from_support(support_rows, table.features, budget.max_input_features)
    xs = pre.transform(support_rows)
    development_rows = [table.rows[int(i)] for i in splits["development"]]
    xd = pre.transform(development_rows)
    yd = table.labels(splits["development"])
    # Reserve one final test evaluation before spending any development budget.
    test_count = len(splits["test"])
    if len(yd) + test_count > budget.max_evaluated_rows:
        raise ValueError("Row budget must cover at least one development pass and final test")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "iterations").mkdir()
    identity = deepcopy(editor.identity)
    manifest = {
        "format_version": 1, "goal": table.goal.to_dict(), "budget": asdict(budget),
        "editor": identity, "features": [asdict(f) for f in table.features],
        "preprocessing": pre.state, "encoded_features": pre.names, "input_size": len(pre.names),
        "classes": classes, "dataset_hash": table.fingerprint(), "split_seed": seed,
        "split_indices": {k: v.tolist() for k, v in splits.items()}, "group_column": table.group_column,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__), "mlsurgie": "0.1.0"},
        "protocol": "direct_weight_programming; support-only preprocessing; final test isolated from editor",
    }
    write_json(output / "manifest.json", manifest)
    support_examples = []
    for j, row in enumerate(support_rows[:budget.support_examples]):
        support_examples.append({"features": {f.name: row[f.name] for f in table.features},
                                 "encoded": xs[j].tolist(), "label": int(str(row[table.goal.target]) == table.goal.positive_label)})
    context_base = {
        "goal": table.goal.to_dict(), "limits": asdict(budget), "program_format": PROGRAM_GUIDE,
        "encoded_features": pre.names, "preprocessing": pre.state, "classes": classes,
        "support_count": len(support_rows), "support_examples": support_examples,
        "support_positive_fraction": float(table.labels(splits["support"]).mean()),
        "feedback_role": "development, used adaptively; final test is inaccessible",
    }
    best = None
    best_metrics = None
    best_iteration = None
    frozen = None
    history = []
    feedback = None
    rows_used = 0
    tokens_used = 0
    stop_reason = "iteration_budget"
    started = time.monotonic()
    for iteration in range(1, budget.iterations + 1):
        remaining = budget.max_seconds - (time.monotonic() - started)
        if remaining <= 0:
            stop_reason = "time_budget"
            break
        if tokens_used >= budget.max_editor_tokens:
            stop_reason = "editor_token_budget"
            break
        if rows_used + len(yd) + test_count > budget.max_evaluated_rows:
            stop_reason = "evaluated_row_budget"
            break
        folder = output / "iterations" / f"{iteration:03d}"
        folder.mkdir()
        context = {**deepcopy(context_base), "iteration": iteration,
                   "best": best.snapshot() if best else None, "last_feedback": deepcopy(feedback),
                   "frozen_hidden_sizes": frozen,
                   "output_token_limit": min(budget.max_output_tokens, budget.max_editor_tokens - tokens_used)}
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
        # Record returned usage even if the submitted program is invalid.
        usage = reply.usage
        used = sum(max(0, int(usage.get(k, 0))) for k in ("input_tokens", "output_tokens"))
        tokens_used += used
        record["usage"] = usage
        if reply.raw is not None:
            (folder / "reply.txt").write_text(reply.raw)
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
            candidate = apply_program(reply.program, len(pre.names), budget, best, frozen)
            rows_used += len(yd)
            probabilities = candidate.probabilities(xd)
            metrics = score(yd, probabilities, table.goal.threshold)
            value = objective(metrics, table.goal.primary_metric)
            delta = save_diff(folder, candidate, best if reply.program["base"] == "best" else None)
            frozen = candidate.hidden_sizes
            selected = best is None or value < objective(best_metrics, table.goal.primary_metric)
            if selected:
                best, best_metrics, best_iteration = candidate, metrics, iteration
            # Show the largest absolute probability errors, not any final-test rows.
            worst = np.argsort(-np.abs(probabilities - yd), kind="stable")[:budget.diagnostic_examples]
            diagnostics = [{"features": {f.name: development_rows[int(i)][f.name] for f in table.features},
                            "encoded": xd[i].tolist(), "label": int(yd[i]), "probability": float(probabilities[i])} for i in worst]
            record.update(status="evaluated", selected=selected, state_hash=candidate.fingerprint(), metrics=metrics, diff=delta)
            feedback = {"status": "evaluated", "metrics": metrics, "selected": selected,
                        "largest_probability_errors": diagnostics}
        except (ValueError, TypeError, KeyError, RuntimeError, OverflowError) as exc:
            record.update(status="invalid", error=f"{type(exc).__name__}: {exc}")
            feedback = {"status": "invalid", "error": record["error"]}
        history.append(record)
        write_json(folder / "result.json", record)
        if best_metrics and table.goal.target_score is not None:
            target = table.goal.target_score
            reached = best_metrics[table.goal.primary_metric] <= target if table.goal.primary_metric in {"log_loss", "brier"} else best_metrics[table.goal.primary_metric] >= target
            if reached:
                stop_reason = "development_target_reached"
                break
    final_metrics = None
    predictor = None
    if best is not None:
        # No more editor calls occur after this point. Freeze before opening test data.
        write_json(output / "selection.json", {"iteration": best_iteration, "state_hash": best.fingerprint(), "primary_metric": table.goal.primary_metric})
        test_rows = [table.rows[int(i)] for i in splits["test"]]
        probabilities = best.probabilities(pre.transform(test_rows))
        final_metrics = score(table.labels(splits["test"]), probabilities, table.goal.threshold)
        rows_used += len(test_rows)
        save_model(output, best, pre, classes, table.goal.threshold, budget)
        predictor = Predictor(best, pre, classes, table.goal.threshold)
    target_met = None
    if final_metrics is not None and table.goal.target_score is not None:
        metric, target = table.goal.primary_metric, table.goal.target_score
        target_met = final_metrics[metric] <= target if metric in {"log_loss", "brier"} else final_metrics[metric] >= target
    report = {
        "status": "no_valid_model" if best is None else "target_not_met" if target_met is False else "completed",
        "editor": identity, "stop_reason": stop_reason, "selected_iteration": best_iteration,
        "development": best_metrics, "final_test": final_metrics, "final_target_met": target_met,
        "history": history, "cost": {"evaluated_rows": rows_used, "editor_tokens": tokens_used,
                                    "wall_seconds": time.monotonic() - started, "currency_cost": None},
        "limitations": ["Adaptive development evaluation can overfit.",
                        "Accuracy interval is Wilson 95%, assumes independent observations; other metrics have no interval here.",
                        "Custom Python editor plugins are trusted; isolation is enforced through built-in JSON editor contexts, not an OS sandbox.",
                        "Token limits stop subsequent work; one provider request can overshoot due to input token usage.",
                        "Scripted fixtures demonstrate the protocol, not autonomous LLM performance."],
    }
    save_report(output, report)
    return BuildResult(output, report, predictor)
