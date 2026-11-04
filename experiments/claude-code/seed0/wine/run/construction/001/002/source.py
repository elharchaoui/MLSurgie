import numpy as np, pandas as pd, json
from bench import support, F, TASK
raw, enc = support(); y = raw.label.values
print('classes', TASK['classes'], 'counts', np.bincount(y))
cols = [f for f in F if f.endswith(':standardized')]
X = enc[cols].values
mu = np.array([X[y == k].mean(0) for k in range(3)]); sd = np.array([X[y == k].std(0) for k in range(3)])
pooled = np.sqrt(sum(((y == k).sum() * sd[k]**2) for k in range(3)) / len(y))
sep = (mu.max(0) - mu.min(0)) / pooled
for i in np.argsort(-sep):
    print(f'{cols[i][:-13]:30s} sep={sep[i]:.2f} means={np.round(mu[:, i], 2)} sds={np.round(sd[:, i], 2)} pooled={pooled[i]:.2f}')
top = [cols[i] for i in np.argsort(-sep)[:6]]
W = sum(pd.DataFrame(X[y == k][:, [cols.index(c) for c in top]], columns=[c[:12] for c in top]).corr() * (y == k).sum() for k in range(3)) / len(y)
print('pooled within-class correlation:'); print(W.round(2).to_string())
