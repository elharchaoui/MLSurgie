import json, numpy as np
s=json.load(open('/inputs/state.json')); print('Best hash',s['best']['hash']); print('Best metrics',s['best_metrics']); f=s['last_feedback']; print('Feedback keys',f.keys());
for k,v in f.items():
 if isinstance(v,list):
  print(k)
  for r in v: print(json.dumps(r))
