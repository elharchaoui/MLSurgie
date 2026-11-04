import json
from pathlib import Path
t=json.loads(Path('/inputs/task.json').read_text()); s=json.loads(Path('/inputs/state.json').read_text())
print('TASK',json.dumps({k:v for k,v in t.items() if k not in ['program_format','preprocessing']},indent=2));print('STATE KEYS',s.keys()); print('BEST KEYS',s['best'].keys())