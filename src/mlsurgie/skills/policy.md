# Policy: reasoning versus fitting

Weight-space programming means the weights come from *your reasoning*. Data analysis is
encouraged; parameter estimation by optimization is not.

## Allowed
- Descriptive statistics: counts, rates, means, medians, quantiles, variances, histograms,
  cross-tabulations, conditional rates in bins, correlations.
- Analytic formulas you derive and can explain: log-odds of a rate, a midpoint between two class
  means, a slope read from two segment means, a centroid you chose from a density plot of bins.
- Computing your own candidate's metric on support rows (or a held-out support slice) to check it.
- Revising a mechanism after reading development feedback, one idea at a time.

## Not allowed
- Optimizers, gradient steps, autodiff, learning libraries (sklearn, torch, statsmodels, ...).
- Solvers that estimate parameters from data: `lstsq`, `solve`, `inv`, `pinv`, `polyfit`,
  `curve_fit`, `scipy.optimize`, iterative k-means/EM updates.
- Search loops that score many weight settings against labels and keep the best
  (grid search, random search, coordinate descent) — that is fitting with extra steps.
- Loading pretrained weights for the target.

## Enforcement
Every executed source and every file in `/workspace` is statically audited each turn.
On the strict track a flagged candidate is rejected (it still costs an iteration); delete or
rewrite the flagged code. On the open track the result is labeled. Transcripts are retained.
