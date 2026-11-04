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

    def _required(self):
        names = [f.name for f in self.features]
        extra = [self.goal.target or self.goal.reference_column, self.group_column]
        return names + [c for c in extra if c]

    def validate(self):
        """Check columns and labels. Returns class names (classification) or None."""
        names = [f.name for f in self.features]
        hidden = {self.goal.target, self.goal.reference_column} - {None}
        if not names or len(set(names)) != len(names) or hidden & set(names):
            raise ValueError("Specify unique input features excluding the target/reference column")
        if not self.rows or any(any(name not in row for name in self._required()) for row in self.rows):
            raise ValueError("Missing required columns or empty dataset")
        if self.group_column and any(row[self.group_column] in (None, "") for row in self.rows):
            raise ValueError("Group identifiers cannot be missing")
        task = self.goal.task
        if task == "regression":
            values = [number(row[self.goal.target]) for row in self.rows]
            if any(v is None for v in values):
                raise ValueError("Regression targets cannot be missing")
            return None
        if task == "clustering":
            if self.goal.reference_column and any(row[self.goal.reference_column] in (None, "") for row in self.rows):
                raise ValueError("Reference labels cannot be missing")
            return None
        labels = {str(row[self.goal.target]) for row in self.rows}
        if "" in labels or "None" in labels:
            raise ValueError("Labels cannot be missing")
        if task == "binary":
            if len(labels) != 2 or self.goal.positive_label not in labels:
                raise ValueError("Exactly two nonmissing labels and a matching positive_label are required")
            return sorted(labels - {self.goal.positive_label}) + [self.goal.positive_label]
        if len(labels) < 3:
            raise ValueError("Multiclass tasks need at least three classes; use task=binary for two")
        return sorted(labels)

    def labels(self, indices):
        """Integer labels (classification), floats (regression), reference ids or None (clustering)."""
        task = self.goal.task
        if task == "binary":
            return np.array([str(self.rows[int(i)][self.goal.target]) == self.goal.positive_label for i in indices], dtype=int)
        if task == "multiclass":
            lookup = {c: k for k, c in enumerate(self.validate())}
            return np.array([lookup[str(self.rows[int(i)][self.goal.target])] for i in indices], dtype=int)
        if task == "regression":
            return np.array([number(self.rows[int(i)][self.goal.target]) for i in indices], dtype=float)
        if not self.goal.reference_column:
            return None
        references = sorted({str(row[self.goal.reference_column]) for row in self.rows})
        lookup = {c: k for k, c in enumerate(references)}
        return np.array([lookup[str(self.rows[int(i)][self.goal.reference_column])] for i in indices], dtype=int)

    def fingerprint(self):
        relevant = self._required()
        values = [{k: row[k] for k in relevant} for row in self.rows]
        return hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode()).hexdigest()


def split_table(table, seed=42):
    """60/20/20; stratified for classification, random otherwise, or disjoint groups."""
    table.validate()
    rng = np.random.default_rng(seed)
    stratified = table.goal.task in {"binary", "multiclass"}
    y = table.labels(range(len(table.rows))) if stratified else None
    def usable(part):
        return len(part) >= 3 and (not stratified or len(set(y[part])) == len(set(y)))
    if table.group_column:
        groups = sorted({str(row[table.group_column]) for row in table.rows})
        if len(groups) < 5:
            raise ValueError("Grouped evaluation needs at least five groups")
        for _ in range(100):
            shuffled = rng.permutation(groups)
            a, b = int(len(groups) * .6), int(len(groups) * .8)
            subsets = [set(shuffled[:a]), set(shuffled[a:b]), set(shuffled[b:])]
            parts = [np.array([i for i, row in enumerate(table.rows) if str(row[table.group_column]) in subset]) for subset in subsets]
            if all(usable(part) for part in parts):
                return dict(zip(("support", "development", "test"), parts))
        raise ValueError("Cannot create three grouped splits containing every class; provide more groups/data")
    if not stratified:
        if len(table.rows) < 15:
            raise ValueError("Need at least fifteen rows for support/development/test")
        order = rng.permutation(len(table.rows))
        a, b = int(len(order) * .6), int(len(order) * .8)
        return {"support": order[:a], "development": order[a:b], "test": order[b:]}
    parts = [[], [], []]
    for label in sorted(set(y.tolist())):
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
