from pathlib import Path
import json
s=json.loads(Path('/inputs/state.json').read_text())
print(json.dumps({k:v for k,v in s.items() if k!='best'},indent=2))
print('BEST keys',list(s.get('best',{})))