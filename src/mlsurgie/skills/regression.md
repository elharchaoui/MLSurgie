# Regression

Output: one value in the target's original units. Selection metric is usually RMSE.

- **Baseline:** output bias = support mean of the target (RMSE = target std). Everything you add must beat it.
- **Linear effects:** a feature with roughly linear conditional mean contributes weight
  = (raw slope) × scale on `x:standardized`. Read the slope from the difference of mean target in
  low vs. high quantile bins divided by the difference of their mean standardized feature.
- **Correlated features:** marginal slopes double-count shared signal. After adding one strong
  feature, look at the residual (target − your prediction) by bins of the next feature and read
  its slope from the residual, not the raw target.
- **Nonlinear shapes:** piecewise-linear via hinges (circuits.md): knots at quantiles where the
  binned mean changes slope.
- **Segments:** a categorical segment shifts the level: weight = segment mean residual.
- **Check:** RMSE and R² on a held-out support slice; plot (in text) mean residual per decile of
  prediction to spot curvature or scale errors.
