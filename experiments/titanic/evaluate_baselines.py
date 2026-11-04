"""Evaluate predeclared baselines only after the WSP run has frozen its model."""
import json
import numpy as np

from run_interactive import OUT, table, splits
from mlsurgie.metrics import score

if not (OUT / 'report.json').exists():
    raise RuntimeError('Finish and freeze the WSP experiment first.')

y = table.labels(range(len(table.rows)))
test = splits['test']
prevalence = float(np.mean(y[splits['support']]))
results = {
    'support_prevalence': score(y[test], np.full(len(test), prevalence)),
    'sex_rule': score(y[test], np.array([
        .75 if table.rows[int(i)]['sex'] == 'female' else .20 for i in test
    ])),
}
(OUT / 'baselines.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2))
