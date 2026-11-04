from pathlib import Path
Path('/workspace/wine_v1.py').write_text(r'''
import numpy as np
from bench import F, FI, support, forward, write
raw, enc = support(); X = enc[F].values; y = raw.label.values; K = 3
# channel -> correlation discount (pooled within-class rho with another chosen channel)
CH = {"flavanoids": 1 / 1.3, "od280_od315_of_diluted_wines": 1.0, "proline": 1.0,
      "color_intensity": 1 / 1.5, "alcohol": 1 / 1.3, "hue": 1 / 1.3}
def build(temperature):
    W = np.zeros((K, len(F))); b = np.log(np.bincount(y, minlength=K) / len(y))
    for f, disc in CH.items():
        x = enc[f + ":standardized"].values
        mu = np.array([x[y == k].mean() for k in range(K)])
        var = sum((y == k).sum() * x[y == k].var() for k in range(K)) / len(y)
        W[:, FI[f + ":standardized"]] = disc * mu / var / temperature
        b = b - disc * mu**2 / (2 * var) / temperature
    return [(W, b)]
def evaluate(layers, rows):
    z = forward(layers, [], X[rows]); z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    return float(-np.log(np.clip(p[np.arange(rows.sum()), y[rows]], 1e-7, 1)).mean()), float((p.argmax(1) == y[rows]).mean())
all_rows = np.ones(len(y), bool); half = np.arange(len(y)) % 2 == 0
layers = build(1.0)
print("T=1 support logloss/acc", evaluate(layers, all_rows), "halves", evaluate(layers, half), evaluate(layers, ~half))
''')
import runpy; runpy.run_path('/workspace/wine_v1.py')
