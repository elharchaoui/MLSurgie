"""Conventionally trained reference models under the exact same protocol.

Same split, same support-only preprocessing, same encoded inputs as the weight-space
editor. Baselines are fit on support rows, a small predeclared hyperparameter grid is
chosen on development rows with the task's selection metric, and the chosen model is
scored once on the final test rows. Requires scikit-learn (pip install -e '.[baselines]').
"""
import numpy as np

from .contracts import improves
from .data import Preprocessor, split_table
from .metrics import score, score_clustering, score_multiclass, score_regression


def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError:
        raise RuntimeError("Baselines need scikit-learn: pip install -e '.[baselines]'") from None


def _candidates(task, seed):
    from sklearn.cluster import KMeans
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.mixture import GaussianMixture
    from sklearn.svm import SVC, SVR
    if task in {"binary", "multiclass"}:
        return {
            "prior": [DummyClassifier(strategy="prior")],
            "logistic_regression": [LogisticRegression(C=c, max_iter=5000) for c in (0.1, 1.0, 10.0)],
            "random_forest": [RandomForestClassifier(n_estimators=300, min_samples_leaf=leaf, random_state=seed, n_jobs=-1)
                              for leaf in (1, 5)],
            "svm_rbf": [SVC(C=c, probability=True, random_state=seed) for c in (0.3, 1.0, 3.0)],
        }
    if task == "regression":
        return {
            "mean": [DummyRegressor()],
            "ridge": [Ridge(alpha=a) for a in (0.1, 1.0, 10.0)],
            "random_forest": [RandomForestRegressor(n_estimators=300, min_samples_leaf=leaf, random_state=seed, n_jobs=-1)
                              for leaf in (1, 5)],
            "svr_rbf": [SVR(C=c) for c in (10.0, 100.0, 1000.0)],
        }
    return {
        "kmeans": [KMeans(n_clusters=None, n_init=10, random_state=seed)],
        "gaussian_mixture": [GaussianMixture(n_components=None, random_state=seed)],
    }


def model_size(model):
    """Rough count of stored numbers, for an honest size comparison with constructed networks."""
    if hasattr(model, "estimators_"):
        return int(sum(e.tree_.node_count * 3 for e in model.estimators_))  # feature, threshold, value per node
    if hasattr(model, "support_vectors_"):
        return int(model.support_vectors_.size + np.size(model.dual_coef_) + np.size(model.intercept_))
    if hasattr(model, "coef_"):
        return int(np.size(model.coef_) + np.size(getattr(model, "intercept_", 0)))
    if hasattr(model, "cluster_centers_"):
        return int(model.cluster_centers_.size)
    if hasattr(model, "means_"):
        return int(model.means_.size + model.covariances_.size + model.weights_.size)
    return 1


def _metrics(goal, model, x, y):
    task = goal.task
    if task == "binary":
        return score(y, model.predict_proba(x)[:, 1], goal.threshold)
    if task == "multiclass":
        probabilities = np.zeros((len(x), goal_classes(model)))
        probabilities[:, model.classes_] = model.predict_proba(x)
        return score_multiclass(y, probabilities)
    if task == "regression":
        return score_regression(y, model.predict(x))
    assignment = model.predict(x)
    scores = np.eye(goal.n_clusters)[assignment]
    return score_clustering(x, scores, y)


def goal_classes(model):
    return int(model._mlsurgie_classes)


def run_baselines(table, seed=42, max_input_features=256):
    """Return {name: {"hyperparameters", "development", "final_test", "size"}} on the protocol split."""
    _require_sklearn()
    goal = table.goal
    classes = table.validate()
    splits = split_table(table, seed)
    rows = {k: [table.rows[int(i)] for i in v] for k, v in splits.items()}
    pre = Preprocessor.from_support(rows["support"], table.features, max_input_features)
    x = {k: pre.transform(v).astype(float) for k, v in rows.items()}
    clustering = goal.task == "clustering"
    y = {k: table.labels(v) for k, v in splits.items()}
    metric = goal.primary_metric
    results = {}
    for name, grid in _candidates(goal.task, seed).items():
        best = None
        for model in grid:
            if clustering:
                key = "n_clusters" if "n_clusters" in model.get_params() else "n_components"
                model.set_params(**{key: goal.n_clusters})
                model.fit(x["support"])
            else:
                model.fit(x["support"], y["support"])
            if goal.task == "multiclass":
                model._mlsurgie_classes = len(classes)
            # Clustering development metrics never use reference labels.
            development = _metrics(goal, model, x["development"], None if clustering else y["development"])
            if best is None or improves(metric, development[metric], best[1][metric]):
                best = (model, development)
        model, development = best
        params = {k: v for k, v in model.get_params().items() if k in {"C", "alpha", "min_samples_leaf", "n_clusters", "n_components"}}
        results[name] = {"hyperparameters": params, "development": development,
                         "final_test": _metrics(goal, model, x["test"], y["test"]), "size": model_size(model)}
    return results
