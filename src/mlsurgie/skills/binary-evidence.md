# Binary classification: building log-odds by hand

Target: a logit z(x) such that sigmoid(z) is a calibrated probability.

1. **Prior.** Output bias b0 = log(p / (1 - p)) with p the support positive rate.
2. **Additive evidence.** For a one-hot category value v: weight = log-odds(rate | v) - log-odds(p),
   shrunk toward 0 when the cell is small (e.g. add a pseudo-count of a few prior-rate examples).
   This is a "weight of evidence" reading of the data, done by you.
3. **Numeric features.** Bin into quantiles, read the rate per bin, and choose a shape: linear
   slope (difference of bin log-odds per standardized unit), hinge, or step (see circuits.md).
4. **Correlated evidence.** Adding independent-looking evidence from correlated features double-counts.
   Shrink overlapping weights (e.g. multiply by 0.5–0.7) or encode the joint cell with an AND unit.
5. **Interactions.** When a rate table by (A, B) is not additive in log-odds, add an AND unit for the
   anomalous cell with output weight equal to the residual log-odds of that cell.
6. **Calibration check.** Compare mean predicted probability with observed rate in deciles of your
   score on support. Over-confident? Scale all output weights by a factor < 1.

Metrics: log loss rewards calibration; ROC-AUC only ranking; accuracy depends on the threshold.
