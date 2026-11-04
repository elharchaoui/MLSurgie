"""Declarative task, feature, and budget contracts shared by every editor."""
import math
from dataclasses import asdict, dataclass

TASKS = ("binary", "multiclass", "regression", "clustering")

# Metrics available for selection, per task. Direction is in LOWER_IS_BETTER.
METRICS = {
    "binary": ("log_loss", "brier", "roc_auc", "average_precision", "accuracy", "balanced_accuracy", "f1"),
    "multiclass": ("log_loss", "accuracy", "balanced_accuracy", "macro_f1"),
    "regression": ("rmse", "mae", "r2"),
    # Clustering selection must not use reference labels; adjusted_rand is reported only.
    "clustering": ("silhouette", "inertia"),
}
DEFAULT_METRIC = {"binary": "log_loss", "multiclass": "log_loss", "regression": "rmse", "clustering": "silhouette"}
LOWER_IS_BETTER = frozenset({"log_loss", "brier", "rmse", "mae", "inertia"})


def improves(metric, value, reference, min_delta=0.0):
    """True when value beats reference by more than min_delta in the metric's direction."""
    if reference is None:
        return True
    return value < reference - min_delta if metric in LOWER_IS_BETTER else value > reference + min_delta


def reaches(metric, value, target):
    return value <= target if metric in LOWER_IS_BETTER else value >= target


@dataclass(frozen=True)
class Feature:
    name: str
    kind: str = "numeric"
    description: str = ""

    def __post_init__(self):
        if not self.name or self.kind not in {"numeric", "categorical"}:
            raise ValueError("Features need a name and numeric/categorical kind")


@dataclass(frozen=True)
class Goal:
    description: str
    target: str | None = None
    positive_label: str = "1"
    primary_metric: str | None = None
    threshold: float = 0.5
    target_score: float | None = None
    task: str = "binary"
    n_clusters: int | None = None
    # Clustering only: optional ground-truth column, never shown to editors, used for reporting.
    reference_column: str | None = None

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        if not self.description.strip():
            raise ValueError("Goal description is required")
        if self.task != "clustering" and not self.target:
            raise ValueError("Supervised tasks need a target column")
        if self.task == "clustering":
            if self.target:
                raise ValueError("Clustering has no target; use reference_column for evaluation labels")
            if type(self.n_clusters) is not int or self.n_clusters < 2:
                raise ValueError("Clustering needs an integer n_clusters >= 2")
        elif self.n_clusters is not None or self.reference_column is not None:
            raise ValueError("n_clusters/reference_column apply only to clustering")
        if self.primary_metric is None:
            object.__setattr__(self, "primary_metric", DEFAULT_METRIC[self.task])
        if self.primary_metric not in METRICS[self.task]:
            raise ValueError(f"Unsupported selection metric for {self.task}: {self.primary_metric}")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
        if self.target_score is not None and not math.isfinite(self.target_score):
            raise ValueError("target_score must be finite")

    def to_dict(self):
        return asdict(self)


OPTIONAL_LIMITS = frozenset({"max_seconds", "max_editor_tokens"})
# Hang guards, not spend limits: they stop a stuck call, never a working run.
TURN_HANG_GUARD_SECONDS = 3600
EVALUATION_HANG_GUARD_SECONDS = 300


@dataclass(frozen=True)
class Budget:
    iterations: int = 10
    max_parameters: int = 4096
    max_width: int = 64
    max_hidden_layers: int = 2
    max_evaluated_rows: int = 100_000
    # Spend limits are opt-in: None means unlimited. Set them only when you want a cap.
    max_seconds: float | None = None
    max_editor_tokens: int | None = None
    max_output_tokens: int = 8192
    support_examples: int = 64
    diagnostic_examples: int = 8
    max_input_features: int = 256
    max_abs_weight: float = 10_000

    def __post_init__(self):
        for key, value in asdict(self).items():
            if value is None and key in OPTIONAL_LIMITS:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Budget {key} must be positive and finite")
            if key not in {"max_seconds", "max_abs_weight"} and not isinstance(value, int):
                raise ValueError(f"Budget {key} must be an integer")
