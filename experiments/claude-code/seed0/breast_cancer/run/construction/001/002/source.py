import numpy as np, pandas as pd
from bench import support, F
raw, enc = support(); y = raw.label.values
cols = [f for f in F if f.endswith(':standardized')]
X = enc[cols].values
d = (X[y == 1].mean(0) - X[y == 0].mean(0)) / np.sqrt((X[y == 1].var(0) + X[y == 0].var(0)) / 2)
order = np.argsort(-np.abs(d))
print('n', len(y), 'malignant', y.sum(), 'rate', round(y.mean(), 3))
for i in order[:16]:
    print(f'{cols[i][:-13]:28s} d={d[i]:+.2f}  mean_m={X[y==1,i].mean():+.2f} mean_b={X[y==0,i].mean():+.2f} sd_m={X[y==1,i].std():.2f} sd_b={X[y==0,i].std():.2f}')
top = [cols[i] for i in order[:10]]
C = pd.DataFrame(X[:, order[:10]], columns=[c[:-13][:14] for c in top]).corr().round(2)
print(C.to_string())
print('missing flags any:', enc[[f for f in F if f.endswith(':missing')]].values.sum())
