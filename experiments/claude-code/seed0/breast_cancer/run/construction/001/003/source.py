import numpy as np, pandas as pd
from bench import support
raw, enc = support(); y = raw.label.values
ch = ['worst_perimeter', 'worst_concave_points', 'worst_texture', 'mean_texture', 'worst_smoothness', 'worst_symmetry', 'worst_fractal_dimension', 'area_error']
Z = pd.DataFrame({c: enc[c + ':standardized'] for c in ch})
for c in ch:
    a, b = Z[c][y == 1], Z[c][y == 0]
    print(f'{c:24s} d={(a.mean()-b.mean())/np.sqrt((a.var()+b.var())/2):+.2f} mean_m={a.mean():+.2f} mean_b={b.mean():+.2f} sd_m={a.std():.2f} sd_b={b.std():.2f}')
print('within-class correlation (benign):'); print(Z[y == 0].corr().round(2).to_string())
print('within-class correlation (malignant):'); print(Z[y == 1].corr().round(2).to_string())
