
import numpy as np
from bench import F, FI, support, write
raw, enc = support(); X = enc[F].values
BEHAVIOR = ["annual_spend", "visits_per_month", "avg_basket", "days_since_last_visit", "online_share"]
# Regions from histogram gaps, defined once (no reassignment): online share gap 0.55-0.65 -> 0.6;
# offline basket gap $66-115 -> $87; offline small-basket spend trough $1.8k-2.7k -> $2,200.
online = raw.online_share > 0.6
big = ~online & (raw.avg_basket > 87)
high_spend = ~online & ~big & (raw.annual_spend > 2200)
low_spend = ~online & ~big & ~high_spend
regions = {"online_first": online, "large_basket_offline": big, "high_spend_regular": high_spend, "low_spend_small_basket": low_spend}
W = np.zeros((4, len(F))); b = np.zeros(4)
for k, (name, m) in enumerate(regions.items()):
    c = np.zeros(len(F))
    for f in BEHAVIOR:
        c[FI[f + ":standardized"]] = X[m.values, FI[f + ":standardized"]].mean()
    W[k] = 2 * c; b[k] = -c @ c
    print(f"{name:24s} n={m.sum():4d}", {f: round(float(raw[f][m].median()), 2) for f in BEHAVIOR})
scores = X @ W.T + b; a = scores.argmax(1)
region_id = np.select(list(regions.values()), range(4))
print("network vs region agreement", round(float((a == region_id).mean()), 3), "sizes", np.bincount(a, minlength=4).tolist())
def silhouette(Z, lab):
    D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1)); s = []
    for i in range(len(Z)):
        own = lab == lab[i]
        if own.sum() <= 1: s.append(0); continue
        a_ = D[i, own].sum() / (own.sum() - 1); b_ = min(D[i, lab == k].mean() for k in set(lab) if k != lab[i])
        s.append((b_ - a_) / max(a_, b_))
    return float(np.mean(s))
print("support silhouette (all encoded inputs, incl. age)", round(silhouette(X, a), 4))
write([(W, b)], [], "Nearest-centroid layer (score_k = 2 c_k.x - |c_k|^2) over the five standardized purchasing features; age "
      "weight 0 (no structure in its histogram). Centroids are one-pass means of four regions defined by histogram gaps: "
      "online share > 0.6 (bimodal, gap 0.55-0.65) = online-first; else average basket > $87 (empty gap $66-115) = "
      "large-basket offline; else annual spend > $2,200 (trough $1.8-2.7k) = high-spend regulars; else low-spend "
      "small-basket. No reassignment or iterative centroid updates.")
