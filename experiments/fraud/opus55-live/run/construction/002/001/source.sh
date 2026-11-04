python -c "
import json;s=json.load(open('/inputs/state.json'));print(json.dumps(s['last_feedback'])[:3000])"