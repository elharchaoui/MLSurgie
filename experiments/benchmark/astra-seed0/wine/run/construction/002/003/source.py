import pandas as pd,json
s=json.load(open('/inputs/state.json'))
d=pd.DataFrame([{**e['features'],'label':e['label'],'probabilities':e['probabilities']} for e in s['last_feedback']['largest_errors']]); print(d.to_string(index=False))