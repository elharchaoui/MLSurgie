
import numpy as np
from bench import F, FI, support, forward, sigmoid, logloss, write
raw, enc = support(); X = enc[F].values; y = raw.label.values
# channel: (feature, correlation discount, reason)
CH = {"worst_perimeter": (1 / 1.55, "size; within-class rho 0.55 with concave points"),
      "worst_concave_points": (1 / 1.55, "shape irregularity; rho 0.55 with size"),
      "worst_texture": (1.0, "texture; within-class rho ~0 with size and shape"),
      "worst_smoothness": (1 / 1.37, "surface smoothness; rho ~0.37 with concave points")}
W = np.zeros((1, len(F))); b = np.log(y.mean() / (1 - y.mean())); ledger = []
for f, (disc, why) in CH.items():
    x = enc[f + ":standardized"].values
    m1, m0 = x[y == 1].mean(), x[y == 0].mean()
    var = (x[y == 1].var() + x[y == 0].var()) / 2
    slope = disc * (m1 - m0) / var          # Gaussian log-likelihood-ratio slope, discounted
    W[0, FI[f + ":standardized"]] = slope
    b -= slope * (m1 + m0) / 2               # centre at the midpoint of the class means
    ledger.append((f, round(m1, 3), round(m0, 3), round(var, 3), round(slope, 3)))
layers = [(W, [b])]
z = forward(layers, [], X)[:, 0]; p = sigmoid(z)
for l in ledger: print(l)
print("bias", round(b, 3), "support logloss", round(logloss(y, p), 4), "acc", round(((p >= .5) == y).mean(), 4))
for lo, hi in [(0, .1), (.1, .3), (.3, .7), (.7, .9), (.9, 1.01)]:
    m = (p >= lo) & (p < hi); print(f"  p in [{lo},{hi}) n={m.sum():3d} observed={y[m].mean() if m.sum() else float('nan'):.2f} mean_p={p[m].mean() if m.sum() else float('nan'):.2f}")
idx = np.arange(len(y)) % 2 == 0
print("split-half logloss", round(logloss(y[idx], p[idx]), 4), round(logloss(y[~idx], p[~idx]), 4))
write(layers, [], "Linear evidence network: four near-independent diagnostic channels (worst perimeter=size, worst concave points=shape, "
      "worst texture, worst smoothness). Each weight is the equal-variance Gaussian log-likelihood-ratio slope (difference of "
      "class means / pooled within-class variance) centred at the class-mean midpoint, discounted by 1/(1+rho) for within-class "
      "correlation with another channel (size-shape 0.55, smoothness-shape 0.37). Bias = support prior log-odds. "
      "Other 26 measurements are redundant with these channels (r 0.85-0.99) and left at zero.")
