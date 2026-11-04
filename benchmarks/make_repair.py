"""Build the segments_repair fixture: a correct nearest-centroid network with planted faults.

The "deployed" model's centroids come from the reference segments (this is the fixture,
not an agent solution). Two faults are then planted, mimicking a bad manual edit:
  1. cluster 1's bias is inflated, so it absorbs neighbouring customers;
  2. cluster 3's online_share and avg_basket coordinates have flipped signs.
Run: python benchmarks/make_repair.py
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
config = json.loads((ROOT / "tasks" / "segments.json").read_text())
names = [f["name"] for f in config["features"]]
rows = list(csv.DictReader((ROOT / "data" / "segments.csv").open()))
x = np.array([[float(r[n]) for n in names] for r in rows])
z = (x - x.mean(0)) / x.std(0)
segments = sorted({r["segment"] for r in rows})
labels = np.array([segments.index(r["segment"]) for r in rows])
weight, bias = [], []
for k in range(len(segments)):
    c = z[labels == k].mean(0)
    c[names.index("age")] = 0.0  # irrelevant to segments
    row = []
    for value in c:
        row += [round(2 * float(value), 4), 0.0]  # [standardized, missing] inputs per numeric feature
    weight.append(row)
    bias.append(round(-float(c @ c), 4))
bias[1] = round(bias[1] + 6.0, 4)
for feature in ("online_share", "avg_basket"):
    j = 2 * names.index(feature)
    weight[3][j] = -weight[3][j]
program = {"base": "zero", "hidden_sizes": [],
           "rationale": "Deployed nearest-centroid segmentation model (contains planted faults).",
           "operations": [{"op": "set", "tensor": "layers.0.weight", "value": weight},
                          {"op": "set", "tensor": "layers.0.bias", "value": bias}]}
(ROOT / "tasks" / "segments_repair.program.json").write_text(json.dumps(program, indent=2) + "\n")
print("wrote segments_repair.program.json")
