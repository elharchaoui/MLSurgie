import sys,json
sys.path.insert(0,'/workspace')
from bench import load_data,forward
import numpy as np,pandas as pd
from scipy.spatial.distance import cdist
from pathlib import Path
task,raw,X=load_data()
meta=json.loads(Path('/workspace/iteration1_diagnostics.json').read_text())
C=np.asarray(meta['centroids'])
region=np.where(raw.online_share>.55,2,np.where(raw.avg_basket>100,3,np.where(raw.annual_spend>2250,1,0)))
behavior=[i for i,s in enumerate(task['encoded_features']) if s.endswith(':standardized') and not s.startswith('age:')]
print('Seed-group within variances (features spend, visits, basket, recency, online):')
V=np.stack([X[region==k][:,behavior].var(axis=0) for k in range(4)])
print(np.round(V,4));print('Total',V.sum(axis=1).round(4))
# Inspect original, already-evaluated candidate only.
state=json.loads(Path('/inputs/state.json').read_text())
parameters=state['best']['parameters']
print('parameter format:',type(parameters).__name__)
if isinstance(parameters,dict): print('parameter keys:',list(parameters))
y=np.argmax(2*X@C.T-np.sum(C*C,axis=1),axis=1)
D=cdist(X,X)
M=np.column_stack([D[:,y==k].mean(axis=1) for k in range(4)])
sizes=np.bincount(y,minlength=4)
a=M[np.arange(len(y)),y]*sizes[y]/(sizes[y]-1)
M[np.arange(len(y)),y]=np.inf
nearest=M.argmin(axis=1);b=M.min(axis=1)
s=(b-a)/np.maximum(a,b)
idx=np.argsort(s)[:16]
print(raw.iloc[idx].assign(cluster=y[idx],neighbor=nearest[idx],silhouette=s[idx]).round(4).to_string())
print('Baseline per-group a,b means',[(k,round(a[y==k].mean(),3),round(b[y==k].mean(),3)) for k in range(4)])