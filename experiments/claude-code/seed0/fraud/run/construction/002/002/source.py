import numpy as np, pandas as pd
from bench import support
raw, _ = support(); y = raw.label.values
core = (raw.account_age_days < 30) & (raw.card_present == 'no') & raw.merchant_category.isin(['electronics', 'gift_cards'])
print('new&cnp&risky', core.sum(), y[core].sum())
bins = [0, 50, 100, 150, 200, 300, 500, 1e9]
t = pd.DataFrame({'amt': pd.cut(raw.amount[core], bins), 'y': y[core]}).groupby('amt', observed=False).y.agg(['size', 'sum'])
print(t.to_string())
for age in [14, 30, 45, 60, 90]:
    m = (raw.account_age_days < age) & (raw.card_present == 'no') & raw.merchant_category.isin(['electronics', 'gift_cards']) & (raw.amount >= 100)
    print('age<', age, 'cnp risky amt>=100', m.sum(), y[m].sum())
