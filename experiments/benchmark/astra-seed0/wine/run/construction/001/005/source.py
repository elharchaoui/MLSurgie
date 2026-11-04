import pandas as pd, numpy as np, json
raw=pd.read_csv('/inputs/data/support.csv'); enc=pd.read_csv('/inputs/data/support_encoded.csv'); task=json.load(open('/inputs/task.json'))
print('raw',raw.shape,raw.head().to_string(index=False)); print('encoded',enc.shape,enc.head(2).to_string(index=False))
print('COUNTS',raw.label.value_counts().sort_index().to_dict()); print('MISSING',raw.isna().sum().to_dict())
print('RAW MEANS\n',raw.groupby('label').mean().round(3).T.to_string())
print('STANDARDIZED MEANS\n',enc[[c for c in enc if c.endswith(':standardized')]+['label']].groupby('label').mean().round(3).T.to_string())
print('STANDARDIZED SDs\n',enc[[c for c in enc if c.endswith(':standardized')]+['label']].groupby('label').std().round(3).T.to_string())
x=enc[[c for c in enc if c.endswith(':standardized')]]
resid=x-x.groupby(enc.label).transform('mean')
print('WITHIN-CLASS CORRELATIONS\n',resid.corr().round(2).rename(index={c:c.split(':')[0][:10] for c in x},columns={c:c.split(':')[0][:10] for c in x}).to_string())
print('RAW QUANTILES\n',raw.groupby('label').quantile([.05,.25,.5,.75,.95]).round(3).T.to_string())