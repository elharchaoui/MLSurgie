"""Local table handling. Preprocessing statistics use support rows only."""
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contracts import Feature, Goal


@dataclass
class Table:
    rows: list[dict]
    features: list[Feature]
    goal: Goal
    group_column: str | None = None

    @classmethod
    def from_csv(cls, path, features, goal, group_column=None):
        with Path(path).open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            names = reader.fieldnames or []
            if len(names) != len(set(names)):
                raise ValueError("Duplicate CSV column names")
            rows = list(reader)
        return cls(rows, features, goal, group_column)

    def validate(self):
        names = [f.name for f in self.features]
        if not names or len(set(names)) != len(names) or self.goal.target in names:
            raise ValueError("Specify unique input features excluding the target")
        required = names + [self.goal.target] + ([self.group_column] if self.group_column else [])
        if not self.rows or any(any(name not in row for name in required) for row in self.rows):
            raise ValueError("Missing required columns or empty dataset")
        labels = {str(row[self.goal.target]) for row in self.rows}
        if len(labels) != 2 or self.goal.positive_label not in labels or "" in labels or "None" in labels:
            raise ValueError("Exactly two nonmissing labels and a matching positive_label are required")
        if self.group_column and any(row[self.group_column] in (None, "") for row in self.rows):
            raise ValueError("Group identifiers cannot be missing")
        return sorted(labels - {self.goal.positive_label}) + [self.goal.positive_label]

    def labels(self, indices):
        return np.array([str(self.rows[int(i)][self.goal.target]) == self.goal.positive_label for i in indices], dtype=int)

    def fingerprint(self):
        relevant = [f.name for f in self.features] + [self.goal.target]
        if self.group_column:
            relevant.append(self.group_column)
        values = [{k: row[k] for k in relevant} for row in self.rows]
        return hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode()).hexdigest()


def split_table(table, seed=42):
    """60/20/20; stratified or disjoint groups. Refuse unsupported tiny splits."""
    table.validate()
    rng = np.random.default_rng(seed)
    y = table.labels(range(len(table.rows)))
    if table.group_column:
        groups = sorted({str(row[table.group_column]) for row in table.rows})
        if len(groups) < 5:
            raise ValueError("Grouped evaluation needs at least five groups")
        for _ in range(100):
            shuffled = rng.permutation(groups)
            a, b = int(len(groups) * .6), int(len(groups) * .8)
            subsets = [set(shuffled[:a]), set(shuffled[a:b]), set(shuffled[b:])]
            parts = [np.array([i for i, row in enumerate(table.rows) if str(row[table.group_column]) in subset]) for subset in subsets]
            if all(len(part) and len(set(y[part])) == 2 for part in parts):
                return dict(zip(("support", "development", "test"), parts))
        raise ValueError("Cannot create three grouped splits containing both classes; provide more groups/data")
    parts = [[], [], []]
    for label in (0, 1):
        indices = rng.permutation(np.flatnonzero(y == label))
        if len(indices) < 5:
            raise ValueError("Need at least five examples per class for support/development/test")
        a, b = int(len(indices) * .6), int(len(indices) * .8)
        for out, part in zip(parts, (indices[:a], indices[a:b], indices[b:])):
            out.extend(part.tolist())
    return {name: rng.permutation(part) for name, part in zip(("support", "development", "test"), parts)}


def number(value):
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid numeric feature value: {value!r}") from exc
    if not np.isfinite(result):
        raise ValueError("Numeric feature values must be finite or empty")
    return result


class Preprocessor:
    def __init__(self, state):
        self.state = state

    @classmethod
    def from_support(cls, rows, features, max_features=256):
        state = []
        for feature in features:
            entry = {"name": feature.name, "kind": feature.kind, "description": feature.description}
            if feature.kind == "numeric":
                values = [number(row[feature.name]) for row in rows]
                present = [v for v in values if v is not None]
                median = float(np.median(present)) if present else 0.
                filled = np.array([median if v is None else v for v in values])
                entry.update(median=median, mean=float(filled.mean()), scale=float(filled.std()) or 1.)
            else:
                entry["categories"] = sorted({str(row[feature.name]) for row in rows if row[feature.name] not in (None, "")})
            state.append(entry)
        result = cls(state)
        if len(result.names) > max_features:
            raise ValueError("Encoded feature count exceeds budget; remove identifiers/high-cardinality features")
        return result

    @property
    def names(self):
        names = []
        for f in self.state:
            if f["kind"] == "numeric":
                names.extend([f["name"] + ":standardized", f["name"] + ":missing"])
            else:
                names.extend([f["name"] + "=" + repr(v) for v in f["categories"]])
                names.extend([f["name"] + ":missing", f["name"] + ":unknown"])
        return names

    def transform(self, rows):
        output = []
        for row in rows:
            vector = []
            for f in self.state:
                if f["name"] not in row:
                    raise ValueError(f"Missing input column {f['name']}")
                raw = row[f["name"]]
                if f["kind"] == "numeric":
                    value = number(raw)
                    vector.extend([((f["median"] if value is None else value) - f["mean"]) / f["scale"], float(value is None)])
                else:
                    missing = raw in (None, "")
                    value = str(raw)
                    vector.extend([float(not missing and value == c) for c in f["categories"]])
                    vector.extend([float(missing), float(not missing and value not in f["categories"])])
            output.append(vector)
        array = np.asarray(output, dtype=np.float32).reshape(len(rows), len(self.names))
        if not np.isfinite(array).all():
            raise ValueError("Preprocessing produced nonfinite inputs")
        return array
