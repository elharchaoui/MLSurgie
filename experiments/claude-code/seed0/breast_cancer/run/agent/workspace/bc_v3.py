
import numpy as np
from bench import F, FI, support, forward, sigmoid, logloss, write
raw, enc = support(); X = enc[F].values; y = raw.label.values
CH = {"worst_perimeter": 1 / 1.55, "worst_concave_points": 1 / 1.55, "worst_texture": 1.0, "worst_smoothness": 1 / 1.37}
n_in = len(F)
# Hidden layer: identity pass-through of the linear evidence (relu(e) - relu(-e)), plus heteroscedastic hinges.
units_W, units_b, out_w = [], [], []
bias = np.log(y.mean() / (1 - y.mean()))
for f, disc in CH.items():
    x = enc[f + ":standardized"].values
    m1, m0 = x[y == 1].mean(), x[y == 0].mean(); s1, s0 = x[y == 1].std(), x[y == 0].std()
    slope = disc * (m1 - m0) / ((s1**2 + s0**2) / 2)
    e = np.zeros(n_in); e[FI[f + ":standardized"]] = slope
    units_W += [e, -e]; units_b += [-slope * (m1 + m0) / 2, slope * (m1 + m0) / 2]; out_w += [1.0, -1.0]
    # Unequal variances: the exact Gaussian LLR has a quadratic term (1/(2 s0^2) - 1/(2 s1^2)) (x - m0)^2.
    # Beyond the benign range (m0 + 2.5 s0) approximate its growth by a hinge with the LLR's slope at m0 + 3 s0.
    if s1 > 1.5 * s0:
        knot = m0 + 2.5 * s0
        curv = (1 / (2 * s0**2) - 1 / (2 * s1**2))
        hinge_slope = disc * 2 * curv * (3 * s0) * 0.5      # half the local quadratic slope: conservative
        h = np.zeros(n_in); h[FI[f + ":standardized"]] = 1.0
        units_W.append(h); units_b.append(-knot); out_w.append(hinge_slope)
        print("hinge", f, "knot", round(knot, 3), "slope", round(hinge_slope, 3), "sd_m", round(s1, 2), "sd_b", round(s0, 2))
# Heterogeneity channel: area_error (nuclear-area variability) is near-constant in benign masses (sd 0.17)
# and very spread in malignant ones (sd 1.38): evidence only above the benign range, as a hinge,
# discounted 1/(1+0.55) for its within-class correlation (0.4-0.7) with size.
x = enc["area_error:standardized"].values
m0, s0, s1 = x[y == 0].mean(), x[y == 0].std(), x[y == 1].std()
knot = m0 + 2.5 * s0
hinge_slope = (1 / 1.55) * 2 * (1 / (2 * s0**2) - 1 / (2 * s1**2)) * (3 * s0) * 0.5
h = np.zeros(n_in); h[FI["area_error:standardized"]] = 1.0
units_W.append(h); units_b.append(-knot); out_w.append(hinge_slope)
print("hinge area_error knot", round(knot, 3), "slope", round(hinge_slope, 3))
W0, b0 = np.array(units_W), np.array(units_b)
layers = [(W0, b0), (np.array([out_w]), [bias])]
z = forward(layers, ["relu"], X)[:, 0]; p = sigmoid(z)
print("support logloss", round(logloss(y, p), 4), "acc", round(((p >= .5) == y).mean(), 4))
for lo, hi in [(0, .1), (.1, .3), (.3, .7), (.7, .9), (.9, 1.01)]:
    m = (p >= lo) & (p < hi); print(f"  p in [{lo},{hi}) n={m.sum():3d} observed={y[m].mean() if m.sum() else float('nan'):.2f} mean_p={p[m].mean() if m.sum() else float('nan'):.2f}")
idx = np.arange(len(y)) % 2 == 0
print("split-half logloss", round(logloss(y[idx], p[idx]), 4), round(logloss(y[~idx], p[~idx]), 4))
write(layers, ["relu"], "v3: adds an area_error heterogeneity hinge (benign sd 0.17 vs malignant 1.38). Same four Gaussian log-likelihood-ratio channels (size, concave points, texture, smoothness; "
      "1/(1+rho) correlation discounts), passed through ReLU pairs, plus hinges for heteroscedastic channels: malignant "
      "spread exceeds benign spread, so the exact Gaussian LLR is quadratic; beyond the benign range (mean+2.5sd) a hinge "
      "adds half the local quadratic slope. Bias = prior log-odds.")
