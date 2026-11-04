import numpy as np, pandas as pd
from bench import support, F
raw, enc = support(); y = raw.target.values
print('n', len(y), 'mean', round(y.mean(), 1), 'sd', round(y.std(), 1))
cols = [f for f in F if f.endswith(':standardized')] + ["sex='2.0'"] if "sex='2.0'" in F else [f for f in F if f.endswith(':standardized')]
print('encoded:', F)
Z = enc[[c for c in F if c.endswith(':standardized') or c.startswith('sex=')]]
r = Z.apply(lambda c: np.corrcoef(c, y)[0, 1])
print('corr with target:'); print(r.round(3).sort_values().to_string())
for c in Z.columns:
    if c.startswith('sex'): continue
    q = pd.qcut(Z[c], 5, duplicates='drop')
    print(f'{c[:-13]:4s} quintile means:', pd.Series(y).groupby(q.values, observed=True).mean().round(0).tolist())
print(Z.corr().round(2).to_string())
