import json
from pathlib import Path
task = json.loads(Path('/inputs/task.json').read_text())
print(json.dumps({k:v for k,v in task.items() if k not in ['preprocessing','program_format']},indent=2))
print('STATE',Path('/inputs/state.json').read_text())