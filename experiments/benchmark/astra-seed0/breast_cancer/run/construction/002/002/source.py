import json,sys,numpy as np,pandas as pd
sys.path.insert(0,'/workspace')
from bench import *
s=json.load(open('/inputs/state.json')); fb=s['last_feedback']
print('Feedback keys',fb.keys())
for k,v in fb.items():
 if isinstance(v,list):
  print('Error rows field',k,len(v))
  for r in v:
   print('label',r['label'],'p',round(r['probability'],4),{f:float(r['features'][f]) for f in ['worst_radius','worst_concave_points','worst_texture','radius_error']})
p=json.load(open('/workspace/candidate_v1.json'));z=forward(p)
print('Worst build rows')
loss=np.logaddexp(0,z)-y*z
for i in build[np.argsort(loss[build])[-10:][::-1]]:
 print(i,'y',y[i],'z',round(z[i],3),raw.loc[i,['worst_radius','worst_concave_points','worst_texture','radius_error']].round(4).to_dict())
print('Scalar distribution quantiles')
score=np.load('/workspace/predictions_v1.npz')['score']
for label in [0,1]:
 t=score[build[y[build]==label]]
 print(label,len(t),np.round(np.quantile(t,[0,.05,.1,.25,.5,.75,.9,.95,1]),3))