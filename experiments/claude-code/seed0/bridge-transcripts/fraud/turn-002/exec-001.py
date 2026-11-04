from bench import state
s = state()
for r in s['last_feedback']['largest_errors']:
    print(r['label'], round(r['probability'], 4), r['features'])
