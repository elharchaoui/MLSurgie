import numpy as np
from bench import F, FI, PRE, support, state
raw, enc = support(); X = enc[F].values
best = state()['best']; W = np.array(best['parameters']['layers.0.weight']); b = np.array(best['parameters']['layers.0.bias'])
print('hash', best['hash'][:12], 'arch', best['hidden_sizes'], 'shape', W.shape)
C = W / 2
feats = [f for f in F if f.endswith(':standardized')]
a = (X @ W.T + b).argmax(1); print('support assignment sizes', np.bincount(a, minlength=4).tolist())
print('check 1: bias vs -|c|^2 (nearest-centroid identity)')
for k in range(4):
    print(f'  cluster {k}: bias={b[k]:+.3f}  -|c|^2={-(C[k] @ C[k]):+.3f}  excess={b[k] + C[k] @ C[k]:+.3f}')
print('check 2: implied centroids in raw units vs observed support range')
for k in range(4):
    row = {}
    for f in feats:
        n = f[:-13]; p = PRE[n]; v = C[k, FI[f]] * p['scale'] + p['mean']
        lo, hi = raw[n].min(), raw[n].max(); flag = '  <-- OUTSIDE observed range' if not lo <= v <= hi else ''
        row[n] = f'{v:.2f}{flag}'
    print(f'  cluster {k}:', row)
print('observed ranges', {n: (round(raw[n].min(), 2), round(raw[n].max(), 2)) for n in [f[:-13] for f in feats]})
