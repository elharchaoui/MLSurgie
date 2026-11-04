import pandas as pd, numpy as np
r=pd.read_csv('/inputs/data/support.csv')
print('Raw head\n',r.head(6).to_string(index=False))
print('Full quantiles\n',r.quantile([0,.1,.25,.5,.75,.9,1]).round(2).to_string())
print('Online >.55 profile (all other variables)\n', r.groupby(r.online_share>.55).agg(['mean','std']).round(2).to_string())
off=r[r.online_share<=.55]
print('Offline spend / visits cross tab\n',pd.crosstab(pd.cut(off.annual_spend,[0,1500,2500,3500,4500,6000,11000]),pd.cut(off.visits_per_month,[0,3,6,9,12,15,30])).to_string())
print('Offline basket x visits\n',pd.crosstab(pd.cut(off.avg_basket,[0,40,80,120,160,200,300,500]),pd.cut(off.visits_per_month,[0,3,6,9,12,15,30])).to_string())