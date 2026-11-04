from pathlib import Path
src = Path('/workspace/fraud_v1.py').read_text()
# 1) bust-out amount step: support is empty in (100, 200); dev feedback showed fraud at $152 -> soft step at $125.
src = src.replace('"big":   step("amount", 180, 0.11, above=True),', '"big":   step("amount", 125, 0.06, above=True),')
# 2) drop the overlapping 3-way fallback unit.
src = src.replace('"bustout3": (["new", "big", "cnp"], 2),\n       ', '')
src = src.replace('''# bustout3 overlaps bustout (risky subset); keep it as a weaker fallback for non-risky merchants.
w2[units.index("bustout3")] = 0.5 * w2[units.index("bustout")]
''', '''# Calibrate each signature analytically with the real soft activations: mean logit over the cell's rows,
# computed without the unit, is lifted to the cell's shrunk log-odds (one formula, no search).
for u, (cs, th) in SIG.items():
    j = units.index(u); w2[j] = 0.0
    m = np.all([hard[c] for c in cs], axis=0)
    z_wo = h2 @ w2 + base
    rate = (y[m].sum() + 1) / (m.sum() + 2)
    w2[j] = (logit(rate) - z_wo[m].mean()) / max(h2[m, j].mean(), 1e-6)
    report.append((u + "_calibrated", int(m.sum()), int(y[m].sum()), round(w2[j], 2), round(rate, 3)))
''')
src = src.replace('/workspace/candidate.json', '/workspace/candidate.json')
src = src.replace('Layer 2: ReLU AND units', 'v2: bust-out amount step moved to $125 (support empty in 100-200; dev showed fraud at $152); signature weights calibrated so each cell mean log-odds equals its shrunk support rate (stops 0.999 overconfidence). Layer 2: ReLU AND units')
Path('/workspace/fraud_v2.py').write_text(src)
import runpy; runpy.run_path('/workspace/fraud_v2.py')
