"""Generate the repair model from independent historical customers (seed 29).

The benchmark sample uses seed 11. No benchmark CSV or reference labels are read.
Historical preprocessing accompanies the model; the protocol rebases its inputs to
support-only preprocessing for each run without changing its raw-input behavior.
Two faults are planted: inflated cluster-1 bias and flipped cluster-3 coordinates.
Run: python benchmarks/make_repair.py
"""
import json
from pathlib import Path

import numpy as np

from mlsurgie.contracts import Feature
from mlsurgie.data import Preprocessor

try:
    from .make_datasets import segments
except ImportError:  # direct script invocation
    from make_datasets import segments

ROOT = Path(__file__).resolve().parent
NAMES = ["annual_spend", "visits_per_month", "avg_basket", "days_since_last_visit", "online_share", "age"]


def make_fixture():
    history = segments(n=4000, seed=29, write_data=False)
    rows = [dict(zip([*NAMES, "segment"], row)) for row in history]
    pre = Preprocessor.from_support(rows, [Feature(name) for name in NAMES])
    z = pre.transform(rows).astype(float)
    labels = np.array([r["segment"] for r in rows])
    weight, bias = [], []
    for label in sorted(set(labels)):
        c = z[labels == label].mean(0)
        c[2 * NAMES.index("age")] = 0.0
        weight.append((2 * c).round(4).tolist())
        bias.append(round(-float(c @ c), 4))
    bias[1] = round(bias[1] + 6.0, 4)
    for feature in ("online_share", "avg_basket"):
        weight[3][2 * NAMES.index(feature)] *= -1
    program = {"base": "zero", "hidden_sizes": [],
               "rationale": "Historical nearest-centroid model (independent sample seed 29; two planted faults).",
               "operations": [{"op": "set", "tensor": "layers.0.weight", "value": weight},
                              {"op": "set", "tensor": "layers.0.bias", "value": bias}]}
    return program, pre.state


if __name__ == "__main__":
    program, preprocessing = make_fixture()
    for name, value in (("segments_repair.program.json", program), ("segments_repair.preprocessing.json", preprocessing)):
        (ROOT / "tasks" / name).write_text(json.dumps(value, indent=2) + "\n")
        print(f"wrote {name}")
