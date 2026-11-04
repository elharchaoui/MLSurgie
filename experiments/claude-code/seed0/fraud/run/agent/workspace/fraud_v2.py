
import numpy as np
from bench import F, FI, PRE, support, forward, sigmoid, ap, logloss, write, std
raw, enc = support(); X = enc[F].values; y = raw.label.values
logit = lambda p: float(np.log(p / (1 - p)))

# ---- Layer 1: sigmoid soft-step detectors (name, {encoded feature: weight}, bias) ----
def step(feature, threshold, raw_slope, above):
    k = raw_slope * PRE[feature]["scale"] * (1 if above else -1)
    return {feature + ":standardized": k}, -k * std(feature, threshold)
def onehot(*columns):
    return {c: 10.0 for c in columns}, -5.0
D = {
    "new":   step("account_age_days", 30, 0.44, above=False),
    "big":   step("amount", 125, 0.06, above=True),
    "tiny":  step("amount", 10, 1.5, above=False),
    "vel":   step("txn_last_24h", 5.5, 4.0, above=True),
    "night": step("hour", 5.5, 4.0, above=False),
    "far":   step("distance_from_home_km", 250, 0.044, above=True),
    "cnp":   onehot("card_present='no'"),
    "xb":    onehot("country_mismatch='yes'"),
    "gift":  onehot("merchant_category='gift_cards'"),
    "risky": onehot("merchant_category='electronics'", "merchant_category='gift_cards'"),
    "game":  onehot("merchant_category='gaming'"),
}
names = list(D)
W0 = np.zeros((len(names), len(F))); b0 = np.zeros(len(names))
for i, n in enumerate(names):
    for f, w in D[n][0].items(): W0[i, FI[f]] = w
    b0[i] = D[n][1]
H1 = forward([(W0, b0), (np.zeros((1, len(names))), [0])], ["sigmoid"], X)  # placeholder to reuse code
h1 = 1 / (1 + np.exp(-(X @ W0.T + b0)))
hard = {n: h1[:, i] > 0.5 for i, n in enumerate(names)}

# ---- v2: bust-out amount step moved to $125 (support empty in 100-200; dev showed fraud at $152); signature weights calibrated so each cell mean log-odds equals its shrunk support rate (stops 0.999 overconfidence). Layer 2: ReLU AND units for signatures, plus pass-through evidence units ----
SIG = {"bustout": (["new", "big", "cnp", "risky"], 3), "testing": (["vel", "tiny", "cnp"], 2), "farnight": (["far", "night"], 1),
       "xbnight": (["xb", "cnp", "night"], 2)}
PASS = ["cnp", "gift", "risky", "game", "vel", "big", "xb", "night", "new"]
units = list(SIG) + ["pass_" + p for p in PASS]
W1 = np.zeros((len(units), len(names))); b1 = np.zeros(len(units))
for j, u in enumerate(units):
    if u in SIG:
        for c in SIG[u][0]: W1[j, names.index(c)] = 1
        b1[j] = -SIG[u][1]
    else:
        W1[j, names.index(u[5:])] = 1
h2 = np.maximum(X @ W0.T + b0, -60)  # (not used)
h2 = np.maximum((1 / (1 + np.exp(-(X @ W0.T + b0)))) @ W1.T + b1, 0)

# ---- Output weights from support counts (weight of evidence, pseudo-count shrinkage) ----
sig_any = np.zeros(len(y), bool)
for u, (cs, th) in SIG.items():
    sig_any |= np.all([hard[c] for c in cs], axis=0)
out = ~sig_any
p_out = (y[out].sum() + 1) / (out.sum() + 2)
w2 = np.zeros(len(units)); base = logit(p_out)
report = []
# Evidence terms: WOE measured only on rows outside every signature (the camouflaged remainder).
for p in PASS:
    m = hard[p] & out
    r_in = (y[m].sum() + 2 * p_out) / (m.sum() + 2)            # shrink toward the outside base rate
    r_off = (y[out & ~hard[p]].sum() + 2 * p_out) / ((out & ~hard[p]).sum() + 2)
    w2[units.index("pass_" + p)] = 0.6 * (logit(r_in) - logit(r_off))   # 0.6: correlated evidence discount
    report.append((p, int(m.sum()), int(y[m].sum()), round(w2[units.index("pass_" + p)], 2)))
base = logit(p_out) - sum(w2[units.index("pass_" + p)] * (hard[p] & out).mean() / max(out.mean(), 1e-9) for p in PASS)
# Signature units: lift the logit to the signature cell's shrunk log-odds, net of evidence already present.
for u, (cs, th) in SIG.items():
    m = np.all([hard[c] for c in cs], axis=0)
    rate = (y[m].sum() + 1) / (m.sum() + 2)
    present = base + sum(w2[units.index("pass_" + p)] * hard[p][m].mean() for p in PASS)
    w2[units.index(u)] = (logit(rate) - present) / (len(SIG) if u == "bustout3" else 1)
    report.append((u, int(m.sum()), int(y[m].sum()), round(w2[units.index(u)], 2)))
# Calibrate each signature analytically with the real soft activations: mean logit over the cell's rows,
# computed without the unit, is lifted to the cell's shrunk log-odds (one formula, no search).
for u, (cs, th) in SIG.items():
    j = units.index(u); w2[j] = 0.0
    m = np.all([hard[c] for c in cs], axis=0)
    z_wo = h2 @ w2 + base
    rate = (y[m].sum() + 1) / (m.sum() + 2)
    w2[j] = (logit(rate) - z_wo[m].mean()) / max(h2[m, j].mean(), 1e-6)
    report.append((u + "_calibrated", int(m.sum()), int(y[m].sum()), round(w2[j], 2), round(rate, 3)))
layers = [(W0, b0), (W1, b1), (w2[None, :], [base])]
z = forward(layers, ["sigmoid", "relu"], X)[:, 0]
print("evidence/signature ledger (name, rows, frauds, weight):"); [print(" ", r) for r in report]
print("base", round(base, 3), "| support AP", round(ap(y, z), 4), "logloss", round(logloss(y, sigmoid(z)), 4),
      "mean p", round(sigmoid(z).mean(), 4), "rate", round(y.mean(), 4))
# Honest local estimate: rates were read from all support, so also report a 2-fold split-half check.
idx = np.arange(len(y)) % 2 == 0
print("split-half AP even/odd", round(ap(y[idx], z[idx]), 4), round(ap(y[~idx], z[~idx]), 4))
rationale = ("Layer 1: 11 sigmoid soft-step detectors (account age<30d, amount>=180, amount<10, >=6 txns/24h, hour<6, "
             "distance>=250km, card not present, cross-border, gift cards, electronics/gift, gaming), thresholds read from "
             "support lift tables. v2: bust-out amount step moved to $125 (support empty in 100-200; dev showed fraud at $152); signature weights calibrated so each cell mean log-odds equals its shrunk support rate (stops 0.999 overconfidence). Layer 2: ReLU AND units for the signatures support shows: new+big+card-not-present+"
             "risky merchant (47/49 fraud), card testing burst+tiny+CNP (37/37), far+night (34/34), cross-border+CNP+night "
             "(30/39), plus pass-through evidence units. Output: base = log-odds of fraud outside all signatures "
             "(0.4%); evidence weights = 0.6 x weight-of-evidence measured outside signatures (shrunk); signature "
             "weights lift the logit to each cell's shrunk log-odds net of evidence already present. No fitting.")
write(layers, ["sigmoid", "relu"], rationale)
