"""Task configuration files: goal, features, budget, and protocol components."""
import json
from pathlib import Path

from .contracts import Budget, Feature, Goal
from .data import Table
from .protocol import FeedbackPolicy, SelectionRule, Track
from .tracks import anonymize

ALLOWED = {"name", "notes", "dataset", "goal", "features", "budget", "seed", "group_column",
           "initial_program", "feedback", "selection"}


def load_task(path, dataset=None, track=None):
    """Return (table, options) where options feed protocol.build()."""
    path = Path(path)
    config = json.loads(path.read_text())
    unknown = set(config) - ALLOWED
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    source = dataset or (path.parent / config["dataset"] if config.get("dataset") else None)
    if source is None:
        raise ValueError("Provide --dataset or a `dataset` path in the config")
    goal = Goal(**config["goal"])
    features = [Feature(**f) for f in config["features"]]
    table = Table.from_csv(source, features, goal, config.get("group_column"))
    track = track or Track()
    initial = None
    if config.get("initial_program"):
        initial = json.loads((path.parent / config["initial_program"]).read_text())
    if track.names == "anonymized":
        table = anonymize(table)
    options = {"budget": Budget(**config.get("budget", {})), "seed": config.get("seed", 42),
               "feedback": FeedbackPolicy(**config.get("feedback", {})),
               "selection": SelectionRule(**config.get("selection", {})),
               "track": track, "initial_program": initial}
    return table, options, config
