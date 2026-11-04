import pandas as pd, numpy as np
r=pd.read_csv('/inputs/data/support.csv')
s=r[(r.online_share<.55)&(r.avg_basket<100)]
print('Offline small-basket rows:',len(s))
for name, bins in [('annual_spend',np.arange(500,7501,250)),('visits_per_month',np.arange(0,25,1)),('days_since_last_visit',np.arange(0,34,2))]:
    c,e = np.histogram(s[name],bins=bins)
    print(name, list(zip(e[:-1].tolist(),c.tolist())))
print('Spend x visits within offline small-basket\n',pd.crosstab(pd.cut(s.annual_spend,[0,1500,1750,2000,2250,2500,2750,3000,3500,4500,6500,11000]),pd.cut(s.visits_per_month,[0,4,6,8,10,12,15,30])).to_string())
print('Threshold-defined means:')
region = np.where(r.online_share>.55,'Digital',np.where(r.avg_basket>100,'Premium',np.where(r.annual_spend>2250,'Frequent','Budget')))
print(r.assign(region=region).groupby('region').mean().round(3).to_string())
print(pd.Series(region).value_counts().to_dict())
print('Within-region deviations\n',r.assign(region=region).groupby('region').std().round(3).to_string())