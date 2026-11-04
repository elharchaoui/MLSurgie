"""Task heads: how a network's raw outputs become predictions, metrics, and feedback."""
import numpy as np

from .metrics import score, score_clustering, score_multiclass, score_regression
from .network import sigmoid


def output_size(goal, classes):
    if goal.task == "multiclass":
        return len(classes)
    if goal.task == "clustering":
        return goal.n_clusters
    return 1


def softmax(raw):
    shifted = raw - raw.max(1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(1, keepdims=True)


def head(task, raw):
    """Convert raw outputs [n, k] to the task's prediction representation."""
    raw = np.asarray(raw, dtype=float)
    if task == "binary":
        return sigmoid(raw[:, 0])
    if task == "multiclass":
        return softmax(raw)
    if task == "regression":
        return raw[:, 0]
    return raw  # clustering scores; argmax assigns


def evaluate(goal, y, raw, x):
    prediction = head(goal.task, raw)
    if goal.task == "binary":
        return score(y, prediction, goal.threshold)
    if goal.task == "multiclass":
        return score_multiclass(y, prediction)
    if goal.task == "regression":
        return score_regression(y, prediction)
    return score_clustering(x, prediction, y)


def worst_rows(goal, y, raw, count):
    """Indices and per-row diagnostics of the largest development errors."""
    prediction = head(goal.task, raw)
    if goal.task == "binary":
        error = np.abs(prediction - y)
        detail = lambda i: {"label": int(y[i]), "probability": float(prediction[i])}
    elif goal.task == "multiclass":
        error = 1 - prediction[np.arange(len(y)), y]
        detail = lambda i: {"label": int(y[i]), "probabilities": [round(float(v), 4) for v in prediction[i]]}
    elif goal.task == "regression":
        error = np.abs(prediction - y)
        detail = lambda i: {"target": float(y[i]), "prediction": float(prediction[i])}
    else:
        # Unsupervised: show the least decisive assignments; reference labels are never revealed.
        ordered = np.sort(prediction, 1)
        error = -(ordered[:, -1] - ordered[:, -2]) if prediction.shape[1] > 1 else np.zeros(len(prediction))
        detail = lambda i: {"scores": [round(float(v), 4) for v in prediction[i]], "assigned": int(prediction[i].argmax())}
    order = np.argsort(-error, kind="stable")[:count]
    return [(int(i), detail(int(i))) for i in order]


def reveals_labels(goal):
    return goal.task != "clustering"
