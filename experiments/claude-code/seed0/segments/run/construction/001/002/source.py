import numpy as np
from bench import support
raw, enc = support()
print('n', len(raw), 'columns', list(raw.columns))
for c in raw.columns:
    x = raw[c].values.astype(float)
    v = np.log(x) if c in ('annual_spend', 'visits_per_month', 'avg_basket', 'days_since_last_visit') else x
    h, e = np.histogram(v, 16)
    print(f'{c:22s} {"log" if v is not x else "raw"} range [{e[0]:.2f},{e[-1]:.2f}]', ' '.join(f'{n:3d}' for n in h))
o = raw.online_share > 0.5
print('online>0.5:', o.sum())
for name, m in [('online', o), ('offline', ~o)]:
    sub = raw[m]
    print(name, {c: round(float(np.median(sub[c])), 2) for c in raw.columns})
off = raw[~o]
for c in ['avg_basket', 'visits_per_month', 'annual_spend', 'days_since_last_visit']:
    h, e = np.histogram(np.log(off[c]), 14); print('offline', c, 'log edges', np.round(e[[0, -1]], 2), ' '.join(f'{n:3d}' for n in h))
