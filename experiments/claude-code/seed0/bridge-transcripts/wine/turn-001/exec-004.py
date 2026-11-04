import numpy as np, runpy
g = runpy.run_path('/workspace/wine_v1.py')
X, y, F, layers, enc = g['X'], g['y'], g['F'], g['layers'], g['enc']
from bench import forward
z = forward(layers, [], X); z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
print('confusion (rows true, cols pred):'); C = np.zeros((3, 3), int); np.add.at(C, (y, p.argmax(1)), 1); print(C)
show = ['flavanoids', 'od280_od315_of_diluted_wines', 'proline', 'color_intensity', 'alcohol', 'hue', 'total_phenols', 'malic_acid', 'magnesium', 'alcalinity_of_ash']
for i in np.argsort(p[np.arange(len(y)), y])[:7]:
    print('true', y[i], 'p', np.round(p[i], 3), {s[:10]: round(float(enc[s + ':standardized'][i]), 2) for s in show})
