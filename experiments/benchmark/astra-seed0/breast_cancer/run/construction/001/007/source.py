import json,pandas as pd
p=json.load(open('/inputs/task.json'))
print({k:v for k,v in p.items() if k in ['task','goal','classes','output_size','threshold']})
d=pd.read_csv('/inputs/data/support.csv'); e=pd.read_csv('/inputs/data/support_encoded.csv')
print('shapes',d.shape,e.shape); print(d.head(3).to_string(index=False)); print(d.groupby('label')[['mean_radius','worst_radius','mean_concave_points','worst_concave_points']].agg(['mean','std','count']).round(4).to_string())
print('Missing count',d.isna().sum().sum())