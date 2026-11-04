import sys; sys.path.insert(0,'/workspace')
from bench import load
import numpy as np,pandas as pd
_,_,r,x=load()
offline=(r.online_share<=.6)&(r.avg_basket<=90)
print('Offline low-basket counts',int(offline.sum()))
print('Spend by visits\n',pd.crosstab(pd.cut(r.loc[offline,'annual_spend'],[0,1500,2000,2500,3000,4000,5000,7000,11000]),pd.cut(r.loc[offline,'visits_per_month'],[0,4,6,8,10,14,25])).to_string())
print('Recency by spend\n',pd.crosstab(pd.cut(r.loc[offline,'annual_spend'],[0,1500,2000,2500,3000,4000,5000,7000,11000]),pd.cut(r.loc[offline,'days_since_last_visit'],[0,3,6,9,13,20,120])).to_string())
print('Offline spend fine histogram',list(zip(np.arange(500,7000,250).tolist(),np.histogram(r.loc[offline,'annual_spend'],bins=np.arange(500,7250,250))[0].tolist())))
print('Recency lowspend',r.loc[offline&(r.annual_spend<2500),'days_since_last_visit'].describe().to_dict())
print('Recency highspend',r.loc[offline&(r.annual_spend>=2500),'days_since_last_visit'].describe().to_dict())
