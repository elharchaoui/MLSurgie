import pandas as pd, numpy as np
raw=pd.read_csv('/inputs/data/support.csv'); enc=pd.read_csv('/inputs/data/support_encoded.csv'); x=enc.iloc[:,::2] if False else enc.filter(like=':standardized')
print('counts',raw.label.value_counts().sort_index().to_dict()); print('raw',raw.head(1).to_dict(orient='records')); print('missing?',raw.isna().sum().sum())
print('MEANS standardized\n',x.groupby(enc.label).mean().round(3).T.to_string()); print('STD standardized\n',x.groupby(enc.label).std().round(3).T.to_string())