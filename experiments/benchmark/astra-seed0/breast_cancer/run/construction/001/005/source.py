import json
p=json.load(open('/inputs/task.json'))
for k,v in p.items():
 if k not in ['features','preprocessing','encoded_features','program_format']:
  print(k,':',v)
print('features',p['features'])
print('preprocessing first items',str(p['preprocessing'])[:3000])
print('state',open('/inputs/state.json').read())