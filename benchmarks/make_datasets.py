"""Generate the committed benchmark datasets deterministically.

Synthetic sets have planted structure (known ground truth, absent from any pretraining
corpus). Real sets come from scikit-learn's bundled copies of public UCI datasets.
Run: python benchmarks/make_datasets.py   (requires scikit-learn)
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "data"


def write(name, header, rows):
    OUT.mkdir(exist_ok=True)
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def fraud(n=12000, rate=.02, seed=7):
    """Card transactions: legitimate behavior plus three planted fraud scenarios.

    Scenarios: new-account bust-out (young account, large electronics/gift-card purchase,
    card not present); cross-border card-not-present at night; card testing (bursts of tiny
    gaming/gift-card charges). 20% of fraud is camouflaged as ordinary behavior, so even
    the generating process cannot reach perfect precision.
    """
    rng = np.random.default_rng(seed)
    categories = ["grocery", "retail", "restaurants", "travel", "electronics", "gaming", "gift_cards"]
    label = (rng.random(n) < rate).astype(int)
    scenario = np.where(label == 1, rng.choice(4, n, p=[.3, .25, .25, .2]), -1)  # 3 = camouflaged

    def legit(size):
        category = rng.choice(categories, size, p=[.3, .25, .2, .08, .07, .05, .05])
        amount = np.exp(rng.normal(3.6, 1.0, size)) * np.where(np.isin(category, ["travel", "electronics"]), 3, 1)
        hour = np.clip(np.round(rng.normal(14, 4.5, size)), 0, 23).astype(int)
        night = rng.random(size) < .06
        hour[night] = rng.integers(0, 6, night.sum())
        return {"amount": amount, "hour": hour, "account_age_days": np.round(rng.exponential(900, size)).astype(int),
                "txn_last_24h": rng.poisson(1.5, size), "merchant_category": category,
                "card_present": np.where(rng.random(size) < .7, "yes", "no"),
                "country_mismatch": np.where(rng.random(size) < .05, "yes", "no"),
                "distance_from_home_km": np.exp(rng.normal(2.0, 1.2, size))}

    data = legit(n)
    def put(mask, **columns):
        for key, value in columns.items():
            data[key][mask] = value(mask.sum()) if callable(value) else value
    bust = scenario == 0
    put(bust, account_age_days=lambda k: rng.integers(1, 30, k), amount=lambda k: np.exp(rng.normal(6.2, .5, k)),
        merchant_category=lambda k: rng.choice(["electronics", "gift_cards"], k), card_present="no")
    cross = scenario == 1
    put(cross, card_present="no", country_mismatch=lambda k: np.where(rng.random(k) < .85, "yes", "no"),
        hour=lambda k: rng.integers(0, 6, k), distance_from_home_km=lambda k: np.exp(rng.normal(7.0, .8, k)),
        amount=lambda k: np.exp(rng.normal(5.0, .8, k)))
    testing = scenario == 2
    put(testing, txn_last_24h=lambda k: rng.integers(6, 20, k), amount=lambda k: rng.uniform(.5, 9.99, k),
        merchant_category=lambda k: rng.choice(["gaming", "gift_cards", "retail"], k, p=[.45, .45, .1]), card_present="no")
    amount = np.round(data["amount"], 2)
    distance = np.round(data["distance_from_home_km"], 1).astype(str)
    distance[rng.random(n) < .05] = ""
    rows = zip(amount, data["hour"], data["account_age_days"], data["txn_last_24h"], data["merchant_category"],
               data["card_present"], data["country_mismatch"], distance, label)
    write("fraud.csv", ["amount", "hour", "account_age_days", "txn_last_24h", "merchant_category",
                        "card_present", "country_mismatch", "distance_from_home_km", "is_fraud"], list(rows))
    return scenario


SEGMENTS = {  # log-space means: spend, visits/month, basket, days since last visit; online share
    "bargain_hunters": ([7.2, 1.6, 3.0, 2.3], .15),
    "loyal_regulars": ([8.3, 2.4, 3.5, 1.2], .10),
    "premium_occasional": ([8.6, .3, 5.4, 3.6], .25),
    "online_first": ([7.9, .9, 4.2, 2.6], .85),
}


def segments(n=1600, seed=11, *, write_data=True):
    rng = np.random.default_rng(seed)
    names = list(SEGMENTS)
    which = rng.choice(len(names), n, p=[.3, .3, .15, .25])
    rows = []
    for k in which:
        mean, online = SEGMENTS[names[k]]
        spend, visits, basket, recency = np.exp(rng.normal(mean, [.25, .25, .25, .35]))
        share = float(np.clip(rng.normal(online, .08), 0, 1))
        age = int(np.clip(rng.normal(42, 13), 18, 90))  # unrelated to segment
        rows.append([round(spend, 2), round(visits, 2), round(basket, 2), int(round(recency)), round(share, 3), age, names[k]])
    if write_data:
        write("segments.csv", ["annual_spend", "visits_per_month", "avg_basket", "days_since_last_visit",
                               "online_share", "age", "segment"], rows)
    return rows


def sklearn_sets():
    from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
    data = load_breast_cancer(as_frame=True).frame
    names = [c for c in data.columns if c != "target"]
    write("breast_cancer.csv", [c.replace(" ", "_") for c in names] + ["diagnosis"],
          [[*np.round(r[names].to_numpy(float), 6), "benign" if r["target"] == 1 else "malignant"] for _, r in data.iterrows()])
    data = load_wine(as_frame=True).frame
    names = [c for c in data.columns if c != "target"]
    write("wine.csv", [c.replace("/", "_") for c in names] + ["cultivar"],
          [[*np.round(r[names].to_numpy(float), 6), f"cultivar_{int(r['target']) + 1}"] for _, r in data.iterrows()])
    data = load_diabetes(as_frame=True, scaled=False).frame
    names = [c for c in data.columns if c != "target"]
    write("diabetes.csv", names + ["progression"],
          [[*np.round(r[names].to_numpy(float), 6), r["target"]] for _, r in data.iterrows()])


if __name__ == "__main__":
    fraud()
    segments()
    sklearn_sets()
    checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob("*.csv"))}
    (OUT / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
