from pathlib import Path
Path('/workspace/repair.py').write_text(r'''
import json
import numpy as np
from bench import F, FI, support, state
raw, enc = support(); X = enc[F].values
best = state()["best"]; W = np.array(best["parameters"]["layers.0.weight"]); b = np.array(best["parameters"]["layers.0.bias"])
patch = []
# Fault 1 (check 1): cluster 1 bias exceeds the nearest-centroid identity by +6.2 (others within 0.33).
c1 = W[1] / 2; new_b1 = float(-(c1 @ c1)); patch.append(("layers.0.bias", [1], new_b1, "bias = -|c_1|^2"))
# Fault 2 (check 2): cluster 3 implied avg_basket is negative (impossible) -> sign flip.
j = FI["avg_basket:standardized"]; patch.append(("layers.0.weight", [3, j], float(-W[3, j]), "flip impossible negative basket"))
W2, b2 = W.copy(), b.copy(); b2[1] = new_b1; W2[3, j] = -W[3, j]
# Check 3: implied coordinate vs mean of assigned support rows, testing only the sign-flip hypothesis.
a = (X @ W2.T + b2).argmax(1)
print("sizes after fixes 1-2", np.bincount(a, minlength=4).tolist())
for k in range(4):
    for f in [f for f in F if f.endswith(":standardized") and not f.startswith("age")]:
        c = W2[k, FI[f]] / 2; m = X[a == k, FI[f]].mean(); s = X[a == k, FI[f]].std()
        if abs(c - m) > 1.0 * max(s, 0.25):
            flip_fits = abs(-c - m) < 0.5 * max(s, 0.25)
            print(f"  cluster {k} {f}: implied {c:+.2f}, members' mean {m:+.2f} (sd {s:.2f}) -> {'SIGN FLIP fits' if flip_fits else 'inconsistent, no discrete explanation'}")
            if flip_fits:
                patch.append(("layers.0.weight", [k, FI[f]], float(-W2[k, FI[f]]), f"flip {f} sign (members' mean {m:+.2f})"))
for p in patch: print("PATCH", p)
ops = {}
for tensor, idx, val, why in patch:
    ops.setdefault(tensor, []).append({"index": idx, "value": round(val, 6)})
cand = {"base": "best", "base_hash": best["hash"], "hidden_sizes": [],
        "rationale": "Minimal repair of the nearest-centroid head by three diagnostics. (1) Identity check bias_k = -|W_k/2|^2: "
                     "cluster 1 exceeded it by +6.22 (others within 0.33) and absorbed 615/960 support rows -> bias reset to "
                     "-|c_1|^2. (2) Plausibility: cluster 3 implied average basket -$100 (support minimum $9.61) -> sign "
                     "flipped. (3) Implied coordinate vs mean of its assigned support rows, testing only a sign-flip "
                     "hypothesis: " + "; ".join(p[3] for p in patch[2:]) + ". All other entries unchanged.",
        "operations": [{"op": "set_entries", "tensor": t, "entries": e} for t, e in ops.items()]}
json.dump(cand, open("/workspace/candidate.json", "w"))
W3, b3 = W2.copy(), b2.copy()
for tensor, idx, val, why in patch[2:]: W3[idx[0], idx[1]] = val
print("final support sizes", np.bincount((X @ W3.T + b3).argmax(1), minlength=4).tolist(), "values changed", len(patch))
''')
import runpy; runpy.run_path('/workspace/repair.py')
