import math
import numpy as np


def score(y, probabilities, threshold=.5):
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) == 0 or p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid probability vector")
    prediction = p >= threshold
    tp = int(((y == 1) & prediction).sum())
    tn = int(((y == 0) & ~prediction).sum())
    fp = int(((y == 0) & prediction).sum())
    fn = int(((y == 1) & ~prediction).sum())
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    accuracy = (tp + tn) / len(y)
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    auc = None
    positives, negatives = int(y.sum()), int((1 - y).sum())
    if positives and negatives:
        order = np.argsort(p, kind="stable")
        ranks = np.empty(len(p), dtype=float)
        start = 0
        while start < len(p):
            end = start + 1
            while end < len(p) and p[order[end]] == p[order[start]]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2
            start = end
        auc = float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))
    z, n = 1.96, len(y)
    center = (accuracy + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(accuracy * (1 - accuracy) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {
        "n": n, "positive_count": positives, "threshold": threshold,
        "log_loss": float(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)).mean()),
        "brier": float(((p - y) ** 2).mean()), "roc_auc": auc,
        "accuracy": accuracy, "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
        "precision": precision, "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "accuracy_wilson_95": [max(0., center - radius), min(1., center + radius)],
    }


def objective(metrics, name):
    value = metrics[name]
    if value is None or not np.isfinite(value):
        raise ValueError(f"Metric {name} is undefined")
    return value if name in {"log_loss", "brier"} else -value
