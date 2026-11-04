from pathlib import Path
Path('/workspace/bench.py').write_text(r'''
"""My own bench: data loading, exact IR forward pass, metrics, candidate writer."""
import json
import numpy as np
import pandas as pd

TASK = json.load(open("/inputs/task.json"))
F = TASK["encoded_features"]
FI = {f: i for i, f in enumerate(F)}
PRE = {p["name"]: p for p in TASK["preprocessing"]}
ACT = {"relu": lambda z: np.maximum(z, 0), "tanh": np.tanh,
       "sigmoid": lambda z: 1 / (1 + np.exp(-z)), "identity": lambda z: z}

def state():
    return json.load(open("/inputs/state.json"))

def support():
    raw = pd.read_csv("/inputs/data/support.csv")
    enc = pd.read_csv("/inputs/data/support_encoded.csv")
    return raw, enc

def std(feature, raw_value):
    p = PRE[feature]
    return (raw_value - p["mean"]) / p["scale"]

def forward(layers, acts, X):
    """layers: list of (W [out,in], b [out]); returns raw outputs [n, out]."""
    h = np.asarray(X, dtype=np.float64)
    for i, (W, b) in enumerate(layers):
        h = h @ np.asarray(W).T + np.asarray(b)
        if i < len(layers) - 1:
            h = ACT[acts[i]](h)
    return h

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -60, 60)))

def ap(y, s):
    o = np.argsort(-s, kind="stable"); y, s = np.asarray(y)[o], np.asarray(s)[o]
    d = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[d]; prec = tp / (d + 1); rec = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))

def logloss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7); y = np.asarray(y)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

def write(layers, acts, rationale, path="/workspace/candidate.json"):
    ops = []
    for i, (W, b) in enumerate(layers):
        ops.append({"op": "set", "tensor": f"layers.{i}.weight", "value": np.round(np.asarray(W, float), 6).tolist()})
        ops.append({"op": "set", "tensor": f"layers.{i}.bias", "value": np.round(np.asarray(b, float).reshape(-1), 6).tolist()})
    cand = {"base": "zero", "hidden_sizes": [len(b) for _, b in layers[:-1]], "activations": list(acts),
            "rationale": rationale, "operations": ops}
    json.dump(cand, open(path, "w"))
    return cand
''')
import bench, json
print(open('/inputs/README.md').read()[:400])
print('encoded:', bench.F)
print({k: {x: v[x] for x in ('mean', 'scale', 'median') if x in v} for k, v in bench.PRE.items() if v['kind'] == 'numeric'})
