
import numpy as np
from bench import F, FI, support, forward, write
raw, enc = support(); X = enc[F].values; y = raw.label.values; K = 3
CH = {"flavanoids": 1 / 1.3, "od280_od315_of_diluted_wines": 1.0, "proline": 1.0,
      "color_intensity": 1 / 1.5, "alcohol": 1 / 1.3, "hue": 1 / 1.3}
KNOTS = np.arange(-3.0, 3.01, 0.5)          # x^2 interpolated on these knots
t0, interior = KNOTS[0], KNOTS[1:-1]
s0 = KNOTS[0] + KNOTS[1]                    # slope of the first segment of the x^2 interpolant
units_W, units_b = [], []
out_W = []; out_b = np.log(np.bincount(y, minlength=K) / len(y)); ledger = []
for f, disc in CH.items():
    x = enc[f + ":standardized"].values; col = FI[f + ":standardized"]
    mu = np.array([x[y == k].mean() for k in range(K)])
    pooled = sum((y == k).sum() * x[y == k].var() for k in range(K)) / len(y)
    var = np.array([0.7 * x[y == k].var() + 0.3 * pooled for k in range(K)])   # shrink toward pooled
    q, l, c = 1 / (2 * var), mu / var, mu**2 / (2 * var) + 0.5 * np.log(var)
    def unit(w, b):
        v = np.zeros(len(F)); v[col] = w; units_W.append(v); units_b.append(b)
    unit(1.0, 0.0); unit(-1.0, 0.0)                     # x = relu(x) - relu(-x)
    for t in interior: unit(1.0, -t)                    # hinges building x^2
    ow = np.zeros((K, 2 + len(interior)))
    lin = disc * (-q * s0 + l)
    ow[:, 0], ow[:, 1] = lin, -lin
    ow[:, 2:] = (-disc * q)[:, None]
    out_W.append(ow)
    out_b = out_b + disc * (-q * (t0**2 - s0 * t0) - c)
    ledger.append((f, np.round(mu, 2).tolist(), np.round(np.sqrt(var), 2).tolist()))
W0, b0 = np.array(units_W), np.array(units_b); W1 = np.hstack(out_W)
layers = [(W0, b0), (W1, out_b)]
def evaluate(rows):
    z = forward(layers, ["relu"], X[rows]); z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    return round(float(-np.log(np.clip(p[np.arange(rows.sum()), y[rows]], 1e-7, 1)).mean()), 4), round(float((p.argmax(1) == y[rows]).mean()), 4)
for row in ledger: print(row)
half = np.arange(len(y)) % 2 == 0
print("support", evaluate(np.ones(len(y), bool)), "halves", evaluate(half), evaluate(~half), "units", W0.shape[0])
write(layers, ["relu"], "Per-cultivar diagonal Gaussian log-likelihoods over six chemical channels (flavanoids, OD280, proline, colour "
      "intensity, alcohol, hue). Class means and variances read from support; variances shrunk 30% toward pooled (about 30 "
      "rows per class); within-class correlation discounts 1/(1+rho). The quadratic term -(x-mu_k)^2/(2 sigma_k^2) uses a "
      "ReLU piecewise-linear x^2 (knots every 0.5 on [-3,3]; slope changes by exactly 1 per knot), so the broad cultivar 2 "
      "is no longer pulled into the tight cultivars 1 and 3. Bias = log priors - log sigma terms.")
