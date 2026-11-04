"""Interactive Codex editor experiment; no optimization or parameter fitting."""
import hashlib
import json
from pathlib import Path
import time

from mlsurgie import Budget, Feature, Goal, Table, build
from mlsurgie.artifacts import write_json
from mlsurgie.data import split_table
from mlsurgie.editors import Reply

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/titanic-codex-001'
DATA = ROOT / 'data/titanic-seaborn.csv'
goal = Goal('Construct a neural network predicting Titanic passenger survival by directly programming weights. Optimize development log loss; report accuracy at a fixed 0.5 threshold. No fitting, optimizer, gradients, or test-data inspection.', 'survived')
features = [Feature(n, k, d) for n,k,d in [
    ('pclass','categorical','Ticket class, 1 highest'), ('sex','categorical','Recorded sex'),
    ('age','numeric','Age in years'), ('sibsp','numeric','Siblings/spouses aboard'),
    ('parch','numeric','Parents/children aboard'), ('fare','numeric','Ticket fare'),
    ('embarked','categorical','Embarkation port')]]
table = Table.from_csv(DATA, features, goal)
splits = split_table(table, seed=42)
support = [table.rows[int(i)] for i in splits['support']]

def summary(key):
    groups = {}
    for row in support:
        group = key(row)
        cell = groups.setdefault(group, {'n':0, 'survived':0})
        cell['n'] += 1
        cell['survived'] += int(row['survived'])
    return groups
summaries = {
    'sex_class': summary(lambda r: r['sex'] + '/class' + r['pclass']),
    'sex_age': summary(lambda r: r['sex'] + '/' + ('missing' if not r['age'] else 'under12' if float(r['age'])<12 else '12to17' if float(r['age'])<18 else '18to49' if float(r['age'])<50 else '50plus')),
    'sex_family': summary(lambda r: r['sex'] + '/' + ('alone' if int(r['sibsp'])+int(r['parch'])==0 else 'small1to3' if int(r['sibsp'])+int(r['parch'])<=3 else 'large4plus')),
    'sex_port': summary(lambda r: r['sex'] + '/' + (r['embarked'] or 'missing')),
}

class InteractiveEditor:
    identity = {'kind':'interactive_llm', 'model':'gpt-6-astra', 'interface':'Codex conversation via file bridge', 'token_accounting':'not available from this host session'}
    def propose(self, context, timeout):
        directory = OUT / 'iterations' / f"{context['iteration']:03d}"
        context['support_summaries'] = summaries
        write_json(directory / 'context.json', context)
        request = directory / 'request.json'
        response = directory / 'response.json'
        write_json(request, context)
        print(f'Awaiting weight program: {response}', flush=True)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if response.exists():
                raw = response.read_text()
                return Reply(json.loads(raw), {'input_tokens':0,'output_tokens':0,'measurement':'unavailable'}, raw)
            time.sleep(.2)
        raise TimeoutError('No interactive proposal before deadline')

if __name__ == '__main__':
    result = build(table=table, editor=InteractiveEditor(), output=OUT,
                   budget=Budget(iterations=8, max_seconds=3600, max_parameters=4096,
                                 support_examples=32, diagnostic_examples=12, max_evaluated_rows=2000), seed=42)
    result.report['cost']['editor_tokens'] = None
    result.report['limitations'] += ['Titanic is a public benchmark potentially present in editor pretraining; this is not evidence of uncontaminated learning.', 'IID split; relatives/duplicate passenger-feature rows can cross splits.', 'This interactive host was trusted to avoid test inspection; separation was procedural, not an OS security boundary.', 'Support counts were inspected; no numeric weight-fitting routine was used.']
    from mlsurgie.artifacts import save_report
    save_report(OUT, result.report)
    write_json(OUT / 'dataset_source.json', {'url':'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv', 'source_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(), 'date':'2026-11-04', 'excluded_columns':['alive','class','who','adult_male','deck','embark_town','alone']})
    print(json.dumps(result.report, indent=2), flush=True)
