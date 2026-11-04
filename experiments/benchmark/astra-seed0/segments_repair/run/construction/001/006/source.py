import json,numpy as np
from pathlib import Path
s=json.loads(Path('/inputs/state.json').read_text());t=json.loads(Path('/inputs/task.json').read_text())
print('PREPROCESSING',json.dumps(t['preprocessing'],indent=2)); print('BEST',json.dumps(s['best'],indent=2));print('FEEDBACK',json.dumps(s['last_feedback'],indent=2))