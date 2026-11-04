import numpy as np, pandas as pd
from bench import support
raw, enc = support()
y = raw.label.values; p0 = y.mean()
print('n', len(y), 'positives', y.sum(), 'rate', round(p0, 4))
def lift(mask, name):
    m = np.asarray(mask); n = m.sum()
    if n: print(f'{name:45s} n={n:5d} fraud={y[m].sum():4d} rate={y[m].mean():.3f} share_of_fraud={y[m].sum()/y.sum():.2f}')
for c in ['merchant_category', 'card_present', 'country_mismatch']:
    for v, g in raw.groupby(c): lift(raw[c] == v, f'{c}={v}')
for col, qs in [('amount', [5, 10, 50, 100, 200, 300, 500, 1000]), ('account_age_days', [7, 14, 30, 60, 180]),
                ('txn_last_24h', [2, 3, 4, 5, 6, 8]), ('hour', [3, 6, 8, 20]), ('distance_from_home_km', [50, 250, 500, 1000])]:
    x = raw[col]
    for t in qs:
        lift(x < t, f'{col} < {t}')
        lift(x >= t, f'{col} >= {t}')
