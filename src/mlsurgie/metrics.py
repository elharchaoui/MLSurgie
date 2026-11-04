"""Task metrics implemented in NumPy, so evaluation needs no ML library."""
import math

import numpy as np

from .contracts import LOWER_IS_BETTER


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
    average_precision = average_precision_score(y, p) if positives else None
    n = len(y)
    return {
        "n": n, "positive_count": positives, "threshold": threshold,
        "log_loss": float(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)).mean()),
        "brier": float(((p - y) ** 2).mean()), "roc_auc": auc, "average_precision": average_precision,
        "accuracy": accuracy, "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
        "precision": precision, "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "accuracy_wilson_95": wilson(accuracy, n),
    }


def wilson(rate, n, z=1.96):
    center = (rate + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0., center - radius), min(1., center + radius)]


def average_precision_score(y, p):
    """Step-wise area under the precision-recall curve (ties grouped, as in scikit-learn)."""
    order = np.argsort(-p, kind="stable")
    y, p = y[order], p[order]
    distinct = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]
    tp = np.cumsum(y)[distinct]
    fp = (distinct + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0, recall]) * precision))


def score_multiclass(y, probabilities):
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) == 0 or p.ndim != 2 or p.shape[0] != len(y) or not np.isfinite(p).all():
        raise ValueError("Invalid probability matrix")
    k = p.shape[1]
    prediction = p.argmax(1)
    confusion = np.zeros((k, k), dtype=int)
    np.add.at(confusion, (y, prediction), 1)
    recalls, f1s = [], []
    for c in range(k):
        tp, support, predicted = confusion[c, c], confusion[c].sum(), confusion[:, c].sum()
        if support:
            recalls.append(tp / support)
        f1s.append(2 * tp / (support + predicted) if support + predicted else 0.)
    accuracy = float((prediction == y).mean())
    return {"n": len(y), "classes": k,
            "log_loss": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1)).mean()),
            "accuracy": accuracy, "balanced_accuracy": float(np.mean(recalls)),
            "macro_f1": float(np.mean(f1s)), "confusion_matrix": confusion.tolist(),
            "accuracy_wilson_95": wilson(accuracy, len(y))}


def score_regression(y, prediction):
    y = np.asarray(y, dtype=float)
    f = np.asarray(prediction, dtype=float)
    if len(y) == 0 or f.shape != y.shape or not np.isfinite(f).all():
        raise ValueError("Invalid prediction vector")
    residual = f - y
    variance = float(((y - y.mean()) ** 2).sum())
    return {"n": len(y), "rmse": float(np.sqrt((residual ** 2).mean())), "mae": float(np.abs(residual).mean()),
            "r2": 1 - float((residual ** 2).sum()) / variance if variance else None,
            "target_mean": float(y.mean()), "target_std": float(y.std())}


def adjusted_rand(a, b):
    a, b = np.unique(a, return_inverse=True)[1], np.unique(b, return_inverse=True)[1]
    table = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(table, (a, b), 1)
    pairs = lambda v: (v * (v - 1) / 2).sum()
    index, rows, cols, total = pairs(table), pairs(table.sum(1)), pairs(table.sum(0)), pairs(np.array([len(a)]))
    expected = rows * cols / total if total else 0.
    maximum = (rows + cols) / 2
    return 1. if maximum == expected else float((index - expected) / (maximum - expected))


def silhouette(x, assignment, limit=2000, seed=0):
    """Mean silhouette on encoded inputs (deterministic subsample above `limit` rows)."""
    x, assignment = np.asarray(x, dtype=float), np.asarray(assignment)
    if len(x) > limit:
        keep = np.random.default_rng(seed).choice(len(x), limit, replace=False)
        x, assignment = x[keep], assignment[keep]
    labels = np.unique(assignment)
    if len(labels) < 2:
        return None
    # Keep pairwise work two-dimensional: the broadcast [n, n, features] can
    # exceed 8 GB even within the default feature and sample limits.
    x = x - x.mean(0)  # reduce cancellation in the squared-distance identity
    norms = np.einsum("ij,ij->i", x, x)
    distance = x @ x.T
    distance *= -2
    distance += norms[:, None]
    distance += norms[None, :]
    np.maximum(distance, 0, out=distance)
    np.sqrt(distance, out=distance)
    np.fill_diagonal(distance, 0)
    values = np.zeros(len(x))
    for i in range(len(x)):
        own = assignment == assignment[i]
        if own.sum() <= 1:
            continue  # singleton silhouette is defined as 0
        a = distance[i, own].sum() / (own.sum() - 1)
        b = min(distance[i, assignment == c].mean() for c in labels if c != assignment[i])
        values[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.
    return float(values.mean())


def score_clustering(x, scores, reference=None):
    s = np.asarray(scores, dtype=float)
    if s.ndim != 2 or s.shape[0] != len(x) or not np.isfinite(s).all():
        raise ValueError("Invalid cluster score matrix")
    assignment = s.argmax(1)
    x = np.asarray(x, dtype=float)
    sizes = np.bincount(assignment, minlength=s.shape[1])
    centers = np.array([x[assignment == k].mean(0) if sizes[k] else np.zeros(x.shape[1]) for k in range(s.shape[1])])
    sil = silhouette(x, assignment)
    return {"n": len(x), "clusters_used": int((sizes > 0).sum()), "cluster_sizes": sizes.tolist(),
            "silhouette": sil if sil is not None else -1.0,
            "inertia": float(((x - centers[assignment]) ** 2).sum(1).mean()),
            "adjusted_rand": adjusted_rand(reference, assignment) if reference is not None else None}


def objective(metrics, name):
    value = metrics[name]
    if value is None or not np.isfinite(value):
        raise ValueError(f"Metric {name} is undefined")
    return value if name in LOWER_IS_BETTER else -value
