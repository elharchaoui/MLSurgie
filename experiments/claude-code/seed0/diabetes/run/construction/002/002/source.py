import numpy as np, pandas as pd, runpy
g = runpy.run_path('/workspace/dia_v1.py')
from bench import forward
X, y, F, FI, layers, raw = g['X'], g['y'], g['F'], g['FI'], g['layers'], g['raw']
r = y - forward(layers, [], X)[:, 0]
for f in ['s3', 'bmi', 's5', 'bp', 'age']:
    x = raw[f]; q = pd.qcut(x, 5, duplicates='drop')
    t = pd.DataFrame({'q': q, 'r': r}).groupby('q', observed=True).r.agg(['mean', 'size'])
    print(f, 'residual by quintile:', t['mean'].round(1).tolist(), 'edges', [round(float(i.right), 2) for i in t.index])
