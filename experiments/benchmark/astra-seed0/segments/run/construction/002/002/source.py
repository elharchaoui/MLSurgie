import json
from pathlib import Path
s=json.loads(Path('/inputs/state.json').read_text())
f=s['last_feedback']; print('feedback keys:',list(f));print('best metrics',s['best_metrics'])
for k,v in f.items():
    if not isinstance(v,list): print(k,v)
    else:
        print(k,'length:',len(v))
        for row in v:
            print(row.get('features'), 'assigned:',row.get('assigned'),'scores:',row.get('scores'))
