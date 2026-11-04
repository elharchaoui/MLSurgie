import json
s=json.load(open('/inputs/state.json'))
print('Keys',s.keys()); print('feedback',json.dumps(s.get('last_feedback'),indent=2)[:16000])
print('Best keys',s['best'].keys())