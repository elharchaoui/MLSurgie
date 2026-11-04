
import numpy as np
from bench import F, FI, support, forward, write
raw, enc = support(); X = enc[F].values; y = raw.target.values
def contrast_slope(x, r):
    lo, hi = x <= np.quantile(x, .4), x >= np.quantile(x, .6)
    return (r[hi].mean() - r[lo].mean()) / (x[hi].mean() - x[lo].mean())
ORDER = ["bmi", "s5", "bp", "s3", "sex", "s6", "s4", "age", "s1", "s2"]
w = np.zeros(len(F)); b = y.mean(); ledger = []
pred = np.full(len(y), b)
for f in ORDER:
    col = FI["sex='2.0'"] if f == "sex" else FI[f + ":standardized"]
    x = X[:, col]; r = y - pred
    s = contrast_slope(x, r) if f != "sex" else r[x == 1].mean() - r[x == 0].mean()
    s *= 0.8                                     # shrink each read-off slope (noisy 40%-group contrasts)
    if abs(s) < 3:                               # under 3 target units per SD: not worth the variance
        ledger.append((f, round(s, 2), "dropped")); continue
    w[col] = s; pred = pred + s * (x - x.mean())
    ledger.append((f, round(s, 2), round(float(np.sqrt(((y - pred)**2).mean())), 2)))
b = y.mean() - sum(w[c] * X[:, c].mean() for c in range(len(F)))
layers = [(w[None, :], [b])]
p = forward(layers, [], X)[:, 0]
rmse = lambda m: round(float(np.sqrt(((y[m] - p[m])**2).mean())), 2)
for row in ledger: print(row)
half = np.arange(len(y)) % 2 == 0
print("support RMSE", rmse(np.ones(len(y), bool)), "halves", rmse(half), rmse(~half), "baseline sd", round(y.std(), 1))
write(layers, [], "Additive linear response in target units. Baseline = support mean. Features added once each in order of "
      "strength (BMI, log-triglycerides s5, blood pressure, HDL s3, sex, glucose s6, ...); each slope is read as a two-group "
      "contrast (top 40% minus bottom 40%) of the residual left by previous features, divided by the feature contrast, then "
      "shrunk by 0.8; slopes under 3 units per SD are dropped. No least squares, no refitting passes.")
