import numpy as np
from bench import support
raw, enc = support(); y = raw.label.values
def L(m, name):
    m = np.asarray(m); n = m.sum()
    print(f'{name:52s} n={n:5d} f={y[m].sum():3d} rate={y[m].mean() if n else 0:.3f} share={y[m].sum()/y.sum():.2f}')
for c in ['merchant_category', 'card_present', 'country_mismatch']:
    for v in sorted(raw[c].dropna().unique()): L(raw[c] == v, f'{c}={v}')
a = raw.amount
for t in [10, 100, 180, 300]: L(a < t, f'amount<{t}'); L(a >= t, f'amount>={t}')
cnp = raw.card_present == 'no'; risky = raw.merchant_category.isin(['electronics', 'gift_cards'])
new = raw.account_age_days < 30; big = a >= 180; vel = raw.txn_last_24h >= 6; tiny = a < 10
night = raw.hour < 6; far = raw.distance_from_home_km.fillna(0) >= 250; xb = raw.country_mismatch == 'yes'
L(new & big, 'new&big'); L(new & big & cnp, 'new&big&cnp'); L(new & big & cnp & risky, 'new&big&cnp&risky')
L(vel & tiny, 'vel&tiny'); L(vel & tiny & cnp, 'vel&tiny&cnp'); L(vel, 'vel')
L(far & cnp, 'far&cnp'); L(far & night, 'far&night'); L(xb & cnp, 'xborder&cnp'); L(xb & cnp & night, 'xborder&cnp&night')
L(far & cnp & night, 'far&cnp&night'); L(xb & far, 'xborder&far')
covered = (new & big & cnp) | (vel & tiny) | (far & cnp) | (xb & cnp & night)
L(covered, 'ANY signature'); L(~covered, 'outside signatures')
print('missing distance among fraud', raw.distance_from_home_km[y == 1].isna().sum())
