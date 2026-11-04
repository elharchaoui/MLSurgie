import hashlib
import html
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .contracts import Budget
from .data import Preprocessor
from .network import Network, apply_program


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class Predictor:
    network: Network
    preprocessing: Preprocessor
    classes: list[str]
    threshold: float

    def predict_proba(self, rows):
        p = self.network.probabilities(self.preprocessing.transform(rows))
        return np.column_stack([1 - p, p])

    def predict(self, rows):
        index = (self.predict_proba(rows)[:, 1] >= self.threshold).astype(int)
        return np.asarray(self.classes)[index]


@dataclass
class BuildResult:
    path: Path
    report: dict
    model: Predictor | None


def save_model(path, model, preprocessing, classes, threshold, budget):
    path = Path(path)
    np.savez(path / "weights.npz", **model.arrays())
    write_json(path / "model.json", {
        "format_version": 1, "input_size": model.input_size, "hidden_sizes": model.hidden_sizes,
        "state_hash": model.fingerprint(), "preprocessing": preprocessing.state,
        "classes": classes, "threshold": threshold,
        "limits": {"max_parameters": budget.max_parameters, "max_width": budget.max_width,
                   "max_hidden_layers": budget.max_hidden_layers, "max_abs_weight": budget.max_abs_weight},
    })
    write_json(path / "checksums.json", {name: sha256(path / name) for name in ("weights.npz", "model.json")})


def save_diff(path, candidate, base=None):
    """Sparse numerical changes relative to the program's declared base."""
    before = base.arrays() if base is not None else {}
    changes, total, norm_squared = {}, 0, 0.
    for name, after in candidate.arrays().items():
        old = before.get(name, np.zeros_like(after))
        indices = np.flatnonzero(old.ravel() != after.ravel())
        changes[name + ".indices"] = indices
        changes[name + ".old"] = old.ravel()[indices]
        changes[name + ".new"] = after.ravel()[indices]
        total += len(indices)
        norm_squared += float(np.square(after.astype(float) - old.astype(float)).sum())
    np.savez(Path(path) / "diff.npz", **changes)
    return {"changed_values": total, "l2_change": norm_squared ** .5,
            "base_hash": base.fingerprint() if base is not None else "zero"}


def load(path):
    path = Path(path)
    checksums = read_json(path / "checksums.json")
    for name in ("weights.npz", "model.json"):
        if sha256(path / name) != checksums.get(name):
            raise ValueError(f"Artifact checksum mismatch: {name}")
    config = read_json(path / "model.json")
    if config.get("format_version") != 1:
        raise ValueError("Unsupported artifact version")
    budget = Budget(**config["limits"])
    preprocessing = Preprocessor(config["preprocessing"])
    if config["input_size"] != len(preprocessing.names):
        raise ValueError("Preprocessing/architecture mismatch")
    model = Network(config["input_size"], config["hidden_sizes"], budget)
    with np.load(path / "weights.npz", allow_pickle=False) as arrays, torch.no_grad():
        params = dict(model.named_parameters())
        if set(arrays.files) != set(params):
            raise ValueError("Checkpoint parameter names mismatch")
        for name, param in params.items():
            value = arrays[name]
            if tuple(value.shape) != tuple(param.shape) or value.dtype != np.float32:
                raise ValueError("Checkpoint shape/dtype mismatch")
            if not np.isfinite(value).all() or np.abs(value).max(initial=0) > budget.max_abs_weight:
                raise ValueError("Checkpoint violates value limits")
            param.copy_(torch.from_numpy(value.copy()))
    if model.fingerprint() != config["state_hash"]:
        raise ValueError("Checkpoint state hash mismatch")
    return Predictor(model, preprocessing, config["classes"], config["threshold"])


def replay(path):
    """Reconstruct weights from submitted programs, without inference or an editor."""
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    report = read_json(path / "report.json")
    budget = Budget(**manifest["budget"])
    best = None
    architecture = None
    for record in report["history"]:
        if record["status"] != "evaluated":
            continue
        program = read_json(path / "iterations" / f"{record['iteration']:03d}" / "proposal.json")
        candidate = apply_program(program, manifest["input_size"], budget, best, architecture)
        architecture = candidate.hidden_sizes
        if candidate.fingerprint() != record["state_hash"]:
            raise ValueError("Replay candidate hash mismatch")
        if record["selected"]:
            best = candidate
    if best is None:
        raise ValueError("Run contains no valid selected network")
    if best.fingerprint() != load(path).network.fingerprint():
        raise ValueError("Replay differs from exported checkpoint")
    return best


def save_report(path, report):
    write_json(Path(path) / "report.json", report)
    body = html.escape(json.dumps(report, indent=2, allow_nan=False))
    def cell(value):
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.6g}"
        return html.escape(str(value))
    metric_rows = "".join(
        f"<tr><th>{name}</th><td>{cell((report['development'] or {}).get(name))}</td>"
        f"<td>{cell((report['final_test'] or {}).get(name))}</td></tr>"
        for name in ("n", "log_loss", "brier", "roc_auc", "accuracy", "balanced_accuracy", "precision", "recall", "f1")
    )
    attempts = "".join(
        f"<tr><td>{row['iteration']}</td><td>{cell(row['status'])}</td>"
        f"<td>{cell(row.get('metrics', {}).get('log_loss'))}</td>"
        f"<td>{cell(row.get('diff', {}).get('changed_values'))}</td>"
        f"<td>{'Yes' if row['selected'] else 'No'}</td></tr>"
        for row in report["history"]
    )
    fixture = report["editor"]["kind"] == "scripted_fixture"
    subtitle = "Scripted fixture · verifies the protocol, not LLM reasoning" if fixture else "Live editor run · direct weight programming"
    limitations = "".join(f"<li>{html.escape(item)}</li>" for item in report["limitations"])
    (Path(path) / "report.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>MLSurgie run</title>'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<style>body{max-width:960px;margin:48px auto;padding:0 24px;font:16px/1.6 system-ui;color:#1b2937;}'
        'h1{font-size:36px;letter-spacing:-1px}h2{margin-top:36px}p{max-width:780px}'
        '.tag{font-size:13px;text-transform:uppercase;letter-spacing:1px;color:#146b64}'
        '.summary{background:#edf5f3;border-left:4px solid #146b64;padding:16px 24px}'
        'table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}'
        'th,td{text-align:left;padding:10px;border-bottom:1px solid #dce4e8}'
        'thead{background:#f1f4f6}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f5f7;padding:24px}'
        'details{margin-top:32px}summary{cursor:pointer}li{margin:8px 0}</style>'
        f'<p class="tag">MLSurgie / {html.escape(subtitle)}</p><h1>Weight-space programming run</h1>'
        f'<div class="summary"><strong>{cell(report["status"])}</strong> · selected iteration '
        f'{cell(report["selected_iteration"])}<br>Stop reason: {cell(report["stop_reason"])}</div>'
        '<h2>Prediction quality</h2><p>Development feedback selected the checkpoint. '
        'Final test predictions were computed only after selection.</p>'
        '<table><thead><tr><th>Metric</th><th>Development</th><th>Final test</th></tr></thead>'
        f'<tbody>{metric_rows}</tbody></table>'
        '<h2>Candidate history</h2><p>“Selected” means the candidate became best at that iteration. '
        'Changed values are relative to each proposal’s declared base.</p>'
        '<table><thead><tr><th>Iteration</th><th>Status</th><th>Log loss</th><th>Changed values</th><th>Selected</th></tr></thead>'
        f'<tbody>{attempts}</tbody></table>'
        f'<h2>Compute</h2><p>{cell(report["cost"]["evaluated_rows"])} evaluated rows · '
        f'{cell(report["cost"]["editor_tokens"])} editor tokens · '
        f'{cell(report["cost"]["wall_seconds"])} seconds. Monetary cost is not estimated.</p>'
        f'<h2>Interpretation limits</h2><ul>{limitations}</ul>'
        f'<details><summary>Full metrics, confusion matrices, intervals, and run record</summary><pre>{body}</pre></details></html>'
    )
