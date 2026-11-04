"""Weight-space construction, with no target-model training loop."""

from .contracts import Budget, Feature, Goal
from .data import Table
from .editors import OllamaEditor, ScriptedEditor
from .runner import build
from .artifacts import load, replay

__all__ = ["Budget", "Feature", "Goal", "Table", "OllamaEditor", "ScriptedEditor", "build", "load", "replay"]
__version__ = "0.1.0"
