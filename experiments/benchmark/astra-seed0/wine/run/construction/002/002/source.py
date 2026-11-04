import json
s=json.load(open('/inputs/state.json')); print(s.keys()); print('best',s['best'].keys()); print('feedback',s['last_feedback'].keys());
f=s['last_feedback']; print({k:v for k,v in f.items() if k not in ['largest_errors','errors','diagnostic_examples']});
