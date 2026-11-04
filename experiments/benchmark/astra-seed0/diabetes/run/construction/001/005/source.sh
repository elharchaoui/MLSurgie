python - <<'PY'
import json
j=json.load(open('/inputs/task.json'))
for k,v in j.items():
 if k not in ['program_format','preprocessing']:print(k,':',v)
print('preprocessing:',j['preprocessing'][:4] if isinstance(j['preprocessing'],list) else j['preprocessing'])
PY
cat /skills/network-ir.md