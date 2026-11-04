"""Track transformations. The anonymized track tests how much a result relies on
pretrained knowledge of feature names, descriptions, category values, or the dataset."""
from copy import deepcopy

from .contracts import Feature, Goal
from .data import Table


def anonymize(table):
    """Rename features to f01..., categories to v1..., classes to c1..., drop all descriptions."""
    mapping = {f.name: f"f{i + 1:02d}" for i, f in enumerate(table.features)}
    categories = {}
    for f in table.features:
        if f.kind == "categorical":
            values = sorted({str(r[f.name]) for r in table.rows if r[f.name] not in (None, "")})
            categories[f.name] = {v: f"v{j + 1:0{len(str(len(values)))}d}" for j, v in enumerate(values)}
    goal = table.goal
    target = "target" if goal.target else None
    reference = "reference" if goal.reference_column else None
    labels = {}
    positive = goal.positive_label
    if goal.task in {"binary", "multiclass"}:
        values = sorted({str(r[goal.target]) for r in table.rows})
        labels = {v: f"c{j + 1:0{len(str(len(values)))}d}" for j, v in enumerate(values)}
        positive = labels.get(goal.positive_label, goal.positive_label) if goal.task == "binary" else "1"
    rows = []
    for row in table.rows:
        new = {}
        for name, alias in mapping.items():
            value = row[name]
            new[alias] = categories[name].get(str(value), value) if name in categories else value
        if goal.target:
            new[target] = labels.get(str(row[goal.target]), row[goal.target])
        if goal.reference_column:
            new[reference] = row[goal.reference_column]
        if table.group_column:
            new["group"] = row[table.group_column]
        rows.append(new)
    description = {"binary": "Predict the binary target from anonymized features.",
                   "multiclass": "Predict the class from anonymized features.",
                   "regression": "Predict the numeric target from anonymized features.",
                   "clustering": f"Segment the rows into {goal.n_clusters} clusters from anonymized features."}[goal.task]
    fields = deepcopy(goal.to_dict())
    fields.update(description=description, target=target, reference_column=reference, positive_label=positive)
    features = [Feature(mapping[f.name], f.kind, "") for f in table.features]
    return Table(rows, features, Goal(**fields), "group" if table.group_column else None)
