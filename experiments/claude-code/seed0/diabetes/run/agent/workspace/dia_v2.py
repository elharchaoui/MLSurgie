
import numpy as np, runpy
from bench import F, FI, PRE, forward, write, std
g = runpy.run_path("/workspace/dia_v1.py")
X, y, raw = g["X"], g["y"], g["raw"]
(w, b), = g["layers"]; w = w[0].copy(); b = float(b[0])
r = y - (X @ w + b)
# 1) blood pressure: one more residual contrast (top 40% minus bottom 40%), shrunk 0.8.
x = X[:, FI["bp:standardized"]]
lo, hi = x <= np.quantile(x, .4), x >= np.quantile(x, .6)
dbp = 0.8 * (r[hi].mean() - r[lo].mean()) / (x[hi].mean() - x[lo].mean())
w[FI["bp:standardized"]] += dbp
# 2) HDL threshold: hinge relu(t - s3) with t = standardized 48 mg/dL, sized to the low-HDL residual gap.
s3 = X[:, FI["s3:standardized"]]; t = std("s3", 48.0)
r = y - (X @ w + b)
low = s3 < t
gap = 0.8 * (r[low].mean() - r[~low].mean())
hinge = np.maximum(t - s3, 0)
k = gap / hinge[low].mean()
print("bp extra slope", round(dbp, 2), "| HDL hinge knot (std)", round(t, 3), "gap", round(gap, 2), "slope", round(k, 2))
# network: hidden = [identity pair for the linear part, HDL hinge]; output combines them.
lin = np.zeros(len(F)); lin[:] = w
W0 = np.vstack([lin, -lin, -np.eye(len(F))[FI["s3:standardized"]]])
b0 = np.array([0.0, 0.0, t])
pred_lin = X @ w; bias = float(np.mean(y - pred_lin - k * hinge))
layers = [(W0, b0), (np.array([[1.0, -1.0, k]]), [bias])]
p = forward(layers, ["relu"], X)[:, 0]
rmse = lambda m: round(float(np.sqrt(((y[m] - p[m])**2).mean())), 2)
half = np.arange(len(y)) % 2 == 0
print("support RMSE", rmse(np.ones(len(y), bool)), "halves", rmse(half), rmse(~half))
write(layers, ["relu"], "v2 of the additive response: same residual-contrast slopes (BMI 41, s5 21, BP, HDL, sex, s1 per SD), "
      "plus (a) one further residual contrast for blood pressure, whose residuals still rose across quintiles, and (b) an HDL "
      "threshold hinge relu(48 mg/dL - HDL): low-HDL patients were under-predicted by ~10 units in support residuals and in "
      "development errors; hinge sized to 0.8 x that residual gap. Bias recentred to zero mean residual.")
