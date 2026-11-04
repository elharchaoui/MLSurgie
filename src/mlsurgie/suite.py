"""Benchmark suites: tasks x seeds x one editor, against trained baselines.

Each cell runs the full protocol for the editor and the predeclared baselines on the
identical split and preprocessing, then the suite writes results.json and results.md.
"""
import json
import statistics
from dataclasses import asdict, replace
from pathlib import Path

from .baselines import run_baselines
from .config import load_task
from .contracts import LOWER_IS_BETTER
from .protocol import build

# Metrics reported per task (selection metric first is added dynamically).
REPORTED = {"binary": ["average_precision", "roc_auc", "log_loss", "accuracy"],
            "multiclass": ["accuracy", "macro_f1", "log_loss"],
            "regression": ["rmse", "mae", "r2"],
            "clustering": ["adjusted_rand", "silhouette"]}
# Headline metric used to compare against baselines: external validity for clustering.
HEADLINE = {"clustering": "adjusted_rand"}


def run_suite(suite_path, make_editor, output, *, seeds=None, track=None, editor_options=None, baselines=True):
    """make_editor(task_name, seed, run_directory) -> editor. Returns the summary dict."""
    suite_path = Path(suite_path)
    suite = json.loads(suite_path.read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seeds = suite.get("seeds", [0]) if seeds is None else seeds
    cells = []
    for task_file in suite["tasks"]:
        table, options, config = load_task(suite_path.parent / task_file, track=track)
        name = config.get("name", Path(task_file).stem)
        for seed in seeds:
            directory = output / name / f"seed-{seed}"
            directory.mkdir(parents=True)
            cell = {"task": name, "task_type": table.goal.task, "seed": seed,
                    "selection_metric": table.goal.primary_metric}
            if baselines:
                cell["baselines"] = run_baselines(table, seed)
                (directory / "baselines.json").write_text(json.dumps(cell["baselines"], indent=2, allow_nan=False) + "\n")
            if make_editor is not None:
                editor = make_editor(name, seed, directory / "run")
                try:
                    opts = {**options, "seed": seed, **(editor_options or {})}
                    if "budget_overrides" in opts:
                        opts["budget"] = replace(opts["budget"], **opts.pop("budget_overrides"))
                    result = build(table=table, editor=editor, output=directory / "run", **opts)
                finally:
                    if hasattr(editor, "close"):
                        editor.close()
                report = result.report
                cell["editor"] = {"identity": report["editor"], "status": report["status"],
                                  "final_test": report["final_test"], "development": report["development"],
                                  "parameters": report["parameters"], "iterations": len(report["history"]),
                                  "stop_reason": report["stop_reason"], "cost": report["cost"],
                                  "policy_flags": sum(len((h.get("audit") or {}).get("violations") or []) for h in report["history"])}
            cells.append(cell)
            print(f"[{name} seed {seed}] done", flush=True)
    summary = {"suite": suite.get("name"), "track": asdict(track) if track else None, "seeds": seeds,
               "cells": cells, "table": aggregate(cells)}
    (output / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "results.md").write_text(markdown(summary))
    return summary


def _mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": statistics.fmean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0, "n": len(values)}


def aggregate(cells):
    rows = []
    for task in dict.fromkeys(c["task"] for c in cells):
        group = [c for c in cells if c["task"] == task]
        kind = group[0]["task_type"]
        metrics = list(dict.fromkeys([HEADLINE.get(kind, group[0]["selection_metric"]), *REPORTED[kind]]))
        entry = {"task": task, "task_type": kind, "headline": metrics[0], "methods": {},
                 "editor_kind": (group[0].get("editor") or {}).get("identity", {}).get("kind")}
        names = list(group[0].get("baselines", {}))
        if group[0].get("editor"):
            names = ["editor", *names]
        for method in names:
            def final(c):
                return (c["editor"]["final_test"] if method == "editor" else c["baselines"][method]["final_test"]) or {}
            entry["methods"][method] = {
                "metrics": {m: _mean_sd([final(c).get(m) for c in group]) for m in metrics},
                "size": _mean_sd([(c["editor"]["parameters"] if method == "editor" else c["baselines"][method]["size"]) for c in group]),
            }
        trained = {m: v for m, v in entry["methods"].items() if m not in {"editor", "prior", "mean"}}
        head = metrics[0]
        if trained:
            pick = (min if head in LOWER_IS_BETTER else max)(
                trained, key=lambda m: (trained[m]["metrics"][head] or {"mean": float("nan")})["mean"])
            entry["best_baseline"] = pick
        rows.append(entry)
    return rows


def _fmt(cell):
    if cell is None:
        return "—"
    return f"{cell['mean']:.4f}" + (f" ± {cell['sd']:.4f}" if cell["n"] > 1 else "")


def markdown(summary):
    lines = [f"# Benchmark results: {summary['suite']}", "",
             f"Seeds: {summary['seeds']}. Track: {summary['track']}. Final-test metrics, mean ± sd across seeds.",
             "Baselines are fit on the support split; hyperparameters chosen on development; same preprocessing.", ""]
    for row in summary["table"]:
        metrics = list(next(iter(row["methods"].values()))["metrics"])
        arrow = lambda m: "↓" if m in LOWER_IS_BETTER else "↑"
        lines += [f"## {row['task']} ({row['task_type']})", "",
                  "| Method | " + " | ".join(f"{m} {arrow(m)}" for m in metrics) + " | Stored numbers |",
                  "|---|" + "---:|" * (len(metrics) + 1)]
        for method, values in row["methods"].items():
            label = f"**{row.get('editor_kind') or 'editor'}** (editor)" if method == "editor" else method
            if method == row.get("best_baseline"):
                label += " (best baseline)"
            size = f"{values['size']['mean']:.0f}" if values["size"] else "—"
            lines.append(f"| {label} | " + " | ".join(_fmt(values["metrics"][m]) for m in metrics) + f" | {size} |")
        lines.append("")
    return "\n".join(lines) + "\n"
