import pandas as pd,numpy as np,json
r=pd.read_csv('/inputs/data/support.csv');print(r.head().to_string(index=False));print('SUMMARY\n',r.describe().round(3).to_string());print('MISSING',r.isna().sum().to_dict())
# Counts are descriptive, not scoring any candidate alternatives.
for col,edges in {'annual_spend':[0,1500,2500,3500,5000,7000,11000],'visits_per_month':[0,2,4,6,8,10,14,25],'avg_basket':[0,20,40,60,90,140,200,300,500],'days_since_last_visit':[0,5,10,20,30,45,65,120],'online_share':[0,.1,.2,.3,.5,.7,.9,1.01]}.items():
 print(col, list(zip(edges[:-1],edges[1:],np.histogram(r[col],bins=edges)[0].tolist())))
print('JOINT MODES, basket x visits\n',pd.crosstab(pd.cut(r.avg_basket,[0,40,90,140,500]),pd.cut(r.visits_per_month,[0,2,4,8,25])).to_string());print('JOINT MODES, basket x online\n',pd.crosstab(pd.cut(r.avg_basket,[0,40,90,140,500]),pd.cut(r.online_share,[-.01,.3,.6,1.01])).to_string())