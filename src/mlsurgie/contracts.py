from dataclasses import asdict, dataclass
import math


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
    target: str
    positive_label: str = "1"
    primary_metric: str = "log_loss"
    threshold: float = 0.5
    target_score: float | None = None

    def __post_init__(self):
        if not self.description.strip() or not self.target:
            raise ValueError("Goal description and target are required")
        if self.primary_metric not in {"log_loss", "roc_auc", "accuracy", "balanced_accuracy", "brier"}:
            raise ValueError("Unsupported selection metric")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
        if self.target_score is not None and (not math.isfinite(self.target_score) or self.target_score < 0):
            raise ValueError("target_score must be finite and nonnegative")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Budget:
    iterations: int = 10
    max_parameters: int = 4096
    max_width: int = 64
    max_hidden_layers: int = 2
    max_evaluated_rows: int = 100_000
    max_seconds: float = 600
    max_editor_tokens: int = 100_000
    max_output_tokens: int = 8192
    support_examples: int = 64
    diagnostic_examples: int = 8
    max_input_features: int = 256
    max_abs_weight: float = 10_000

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Budget {key} must be positive and finite")
            if key not in {"max_seconds", "max_abs_weight"} and not isinstance(value, int):
                raise ValueError(f"Budget {key} must be an integer")
