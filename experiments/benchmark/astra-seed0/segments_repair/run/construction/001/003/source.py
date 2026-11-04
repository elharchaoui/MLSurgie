from pathlib import Path
for f in ['/inputs/README.md','/inputs/task.json','/skills/network-ir.md']:
 print('\nFILE',f); print(Path(f).read_text())