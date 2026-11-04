
import pandas as pd, numpy as np
d=pd.read_csv('/inputs/data/support.csv')
print(d.describe(include='all').T)
print(d.label.mean(), d.label.sum())
print(d.isna().mean())
for c in ['merchant_category','card_present','country_mismatch']:
    print(d.groupby(c,dropna=False).label.agg(['mean','sum','count']))
for c in ['amount','hour','account_age_days','txn_last_24h','distance_from_home_km']:
    q=pd.qcut(d[c],10,duplicates='drop') if c!='hour' else d[c]
    print(d.groupby(q,dropna=False).label.agg(['mean','sum','count']))
print(d.groupby(d.distance_from_home_km.isna()).label.agg(['mean','sum','count']))
