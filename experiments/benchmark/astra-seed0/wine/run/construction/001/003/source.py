from pathlib import Path
for f in ['/inputs/README.md','/inputs/task.json','/inputs/state.json']:
 print('\n---'+f+'---\n'+Path(f).read_text())