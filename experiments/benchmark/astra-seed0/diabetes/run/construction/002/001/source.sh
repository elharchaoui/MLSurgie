python - <<'PY'
import json
s=json.load(open('/inputs/state.json'))
print('iteration',s['iteration'])
print('best_metrics',s.get('best_metrics'))
print('last_feedback',json.dumps(s.get('last_feedback'),indent=2))
print('best fields',s.get('best',{}).keys())
PY