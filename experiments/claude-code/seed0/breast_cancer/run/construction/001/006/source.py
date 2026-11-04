from pathlib import Path
src = Path('/workspace/bc_v2.py').read_text()
src = src.replace('''W0, b0 = np.array(units_W), np.array(units_b)''', '''# Heterogeneity channel: area_error (nuclear-area variability) is near-constant in benign masses (sd 0.17)
# and very spread in malignant ones (sd 1.38): evidence only above the benign range, as a hinge,
# discounted 1/(1+0.55) for its within-class correlation (0.4-0.7) with size.
x = enc["area_error:standardized"].values
m0, s0, s1 = x[y == 0].mean(), x[y == 0].std(), x[y == 1].std()
knot = m0 + 2.5 * s0
hinge_slope = (1 / 1.55) * 2 * (1 / (2 * s0**2) - 1 / (2 * s1**2)) * (3 * s0) * 0.5
h = np.zeros(n_in); h[FI["area_error:standardized"]] = 1.0
units_W.append(h); units_b.append(-knot); out_w.append(hinge_slope)
print("hinge area_error knot", round(knot, 3), "slope", round(hinge_slope, 3))
W0, b0 = np.array(units_W), np.array(units_b)''')
src = src.replace('"v2: same four', '"v3: adds an area_error heterogeneity hinge (benign sd 0.17 vs malignant 1.38). Same four')
Path('/workspace/bc_v3.py').write_text(src)
import runpy; runpy.run_path('/workspace/bc_v3.py')
