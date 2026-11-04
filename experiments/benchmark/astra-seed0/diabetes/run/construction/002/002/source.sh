PYTHONPATH=/workspace python - <<'PY'
from bench import *
s=json.loads(Path('/inputs/state.json').read_text())
feedback=s['last_feedback']
print('Feedback keys',list(feedback))
for row in feedback.get('largest_errors',[]):
 print(row)
for key,value in feedback.items():
 if isinstance(value,list):
  print('\n',key)
  for r in value:
   if isinstance(r,dict) and 'features' in r:
    print('y',r['target'],'p',round(r['prediction'],1),'f',r['features'])
program=json.loads(Path('/workspace/candidate_v1.json').read_text())
p=forward(program)
d=RAW.copy();d['p']=p;d['r']=Y-p;d['bin']=pd.qcut(p,10)
print('\nScore deciles\n',d.groupby('bin',observed=True).agg(n=('target','size'),pred=('p','mean'),target=('target','mean'),resid=('r','mean'),sd=('target','std')).round(1).to_string())
print('\nLow score rows\n',d[d.p<80][['target','p','bmi','s5','s3','bp','sex']].sort_values('p').round(2).to_string(index=False))
PY