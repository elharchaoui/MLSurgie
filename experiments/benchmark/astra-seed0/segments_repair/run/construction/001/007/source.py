import json,numpy as np
s=json.load(open('/inputs/state.json'));t=json.load(open('/inputs/task.json'))
print('PREPROCESSING',t['preprocessing']);b=s['best']; print('ARCH', {k:v for k,v in b.items() if k!='parameters'});print('PARAMETERS');print(b['parameters'])