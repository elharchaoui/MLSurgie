from bench import state
for r in state()['last_feedback']['largest_errors']:
    f = r['features']; print(round(r['target']), round(r['prediction']), {k: f[k] for k in ('age', 'sex', 'bmi', 'bp', 's3', 's5', 's1')})
