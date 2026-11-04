"""Static audit of agent-authored code for parameter-fitting signatures.

Weight-space programming allows any analysis (counts, rates, means, quantiles,
histograms, local evaluation of a hand-designed candidate). It forbids estimating
parameters with an optimizer, solver, gradient method, or learning library.

This audit is a heuristic tripwire, not a proof: it catches the obvious routes so that
a "strict" track result means something, and every source file is retained for review.
"""
import re
from pathlib import Path

PATTERNS = [
    (r"\b(import|from)\s+(sklearn|torch|tensorflow|keras|jax|xgboost|lightgbm|catboost|statsmodels|autograd|optax|flax)\b",
     "learning/autodiff library"),
    (r"\bscipy\s*\.\s*optimize\b|\bfrom\s+scipy\s+import\s+[^\n]*\boptimize\b|\bfrom\s+scipy\.optimize\b", "scipy.optimize"),
    (r"\b(linalg\s*\.\s*(lstsq|solve|pinv|inv)|polyfit|curve_fit|minimize|least_squares|nnls)\s*\(",
     "closed-form or numerical solver"),
    (r"\.backward\s*\(|\bgrad(ient)?s?\s*=[^=][^\n]*(\.dot\s*\(|@|\bsum\s*\()", "gradient computation"),
    (r"(-=|\+=)\s*[^\n]*\b(lr|learning_rate|eta|step_size)\b|\b(lr|learning_rate|eta|step_size)\s*\*\s*[^\n]*\bgrad",
     "gradient-style update"),
    (r"\.fit\s*\(|\.partial_fit\s*\(|\bfit_transform\s*\(", "estimator fit call"),
]
COMPILED = [(re.compile(p, re.IGNORECASE), label) for p, label in PATTERNS]
SOURCE_SUFFIXES = {".py", ".sh", ".ipynb"}


def scan_text(text, name):
    findings = []
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for pattern, label in COMPILED:
            if pattern.search(line):
                findings.append({"file": name, "line": number, "rule": label, "text": stripped[:200]})
    return findings


def scan_paths(paths, root=None):
    """Scan files (or directories, recursively) and return audit findings."""
    findings, scanned = [], 0
    for path in paths:
        path = Path(path)
        files = sorted(p for p in path.rglob("*") if p.suffix in SOURCE_SUFFIXES and p.is_file() and not p.is_symlink()) \
            if path.is_dir() else [path] if path.is_file() else []
        for file in files:
            if file.stat().st_size > 2_000_000:
                continue
            scanned += 1
            name = str(file.relative_to(root)) if root and file.is_relative_to(root) else str(file)
            findings.extend(scan_text(file.read_text(errors="replace"), name))
    return {"scanned_files": scanned, "violations": findings}
