from pathlib import Path
import pandas as pd
import numpy as np
pd.set_option('display.max_columns',30)
pd.set_option('display.width',180)
r = pd.read_csv('/inputs/data/support.csv')
x = pd.read_csv('/inputs/data/support_encoded.csv')
print('RAW shape',r.shape, 'ENCODED shape',x.shape)
print(r.head(12).to_string(index=False))
print('ENCODED columns',x.columns.tolist())
print('STATS\n',r.describe(percentiles=[.05,.1,.25,.5,.75,.9,.95]).round(3).to_string())
print('Missing',r.isna().sum().to_dict())
print('Correlations\n',r.corr(numeric_only=True).round(3).to_string())
for name in r.select_dtypes('number'):
    count, edge = np.histogram(r[name].dropna(), bins=20)
    print(name, '\n',[(round(float(a),3),round(float(b),3),int(n)) for a,b,n in zip(edge[:-1],edge[1:],count)])