import json
s=json.load(open('/inputs/state.json')); print(json.dumps(s,indent=2)[:18000])