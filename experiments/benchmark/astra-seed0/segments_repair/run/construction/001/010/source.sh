cat > /workspace/bench.py <<'PY'
"""Local deterministic network execution and descriptive clustering metrics.
No learned parameters, no search. Euclidean metrics use all encoded inputs.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path('/inputs')

def load():
    task = json.loads((ROOT/'task.json').read_text())
    state = json.loads((ROOT/'state.json').read_text())
    raw = pd.read_csv(ROOT/'data/support.csv')
    encoded = pd.read_csv(ROOT/'data/support_encoded.csv')
    x = encoded[task['encoded_features']].to_numpy(dtype=np.float32)
    return task, state, raw, x

def execute(program, task, state):
    widths = [len(task['encoded_features']), *program['hidden_sizes'], task['output_size']]
    if program['base'] == 'best':
        assert program['base_hash'] == state['best']['hash']
        assert program['hidden_sizes'] == state['best']['hidden_sizes']
        parameters = {k:np.array(v,dtype=np.float32) for k,v in state['best']['parameters'].items()}
    else:
        parameters = {}
        for i,(n,m) in enumerate(zip(widths[:-1],widths[1:])):
            parameters[f'layers.{i}.weight'] = np.zeros((m,n),dtype=np.float32)
            parameters[f'layers.{i}.bias'] = np.zeros(m,dtype=np.float32)
    for op in program['operations']:
        a=parameters[op['tensor']]
        if op['op']=='fill':
            a.fill(op['value'])
        elif op['op']=='set':
            a[:]=np.array(op['value'],dtype=np.float32)
        else:
            for item in op['entries']:
                index=tuple(item['index'])
                if op['op']=='set_entries':
                    a[index]=item['value']
                elif op['op']=='add_entries':
                    a[index]+=item['value']
                else:
                    raise ValueError(op['op'])
    return {'hidden_sizes':program['hidden_sizes'],
            'activations':program.get('activations',['relu']*len(program['hidden_sizes'])),
            'parameters':parameters}

def forward(x, network):
    z=np.array(x,dtype=np.float32)
    acts=network.get('activations',['relu']*len(network['hidden_sizes']))
    for i in range(len(network['hidden_sizes'])+1):
        w=np.asarray(network['parameters'][f'layers.{i}.weight'],dtype=np.float32)
        b=np.asarray(network['parameters'][f'layers.{i}.bias'],dtype=np.float32)
        z=z@w.T+b
        if i<len(acts):
            if acts[i]=='relu': z=np.maximum(z,0)
            elif acts[i]=='tanh': z=np.tanh(z)
            elif acts[i]=='sigmoid': z=1/(1+np.exp(-np.clip(z,-80,80)))
            elif acts[i]!='identity': raise ValueError(acts[i])
    return z

def clustering_metrics(x, assignments, outputs=4):
    x=np.asarray(x,dtype=np.float64)
    assignments=np.asarray(assignments)
    sizes=np.bincount(assignments,minlength=outputs)
    used=np.flatnonzero(sizes)
    distances=cdist(x,x,metric='euclidean')
    silhouettes=np.zeros(len(x))
    inertia=0.
    within=np.zeros(len(x))
    nearest=np.full(len(x),np.inf)
    for k in used:
        mask=assignments==k
        center=x[mask].mean(axis=0)
        inertia+=np.square(x[mask]-center).sum()
        mean_dist=distances[:,mask].sum(axis=1)/sizes[k]
        nearest[~mask]=np.minimum(nearest[~mask],mean_dist[~mask])
        if sizes[k]>1:
            within[mask]=mean_dist[mask]*sizes[k]/(sizes[k]-1)
    if len(used)>1:
        silhouettes=(nearest-within)/np.maximum(nearest,within)
        silhouettes[sizes[assignments]==1]=0
    return {'n':len(x),'clusters_used':len(used),'cluster_sizes':sizes.tolist(),
            'silhouette':float(silhouettes.mean()),'inertia':float(inertia/len(x)),
            'silhouette_by_cluster':[float(silhouettes[assignments==k].mean()) if sizes[k] else None for k in range(outputs)]}

def write_candidate(program, path='/workspace/candidate.json'):
    Path(path).write_text(json.dumps(program,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    # Trivial forward-pass invariant: a zero affine network has zero scores.
    task,state,raw,x=load()
    zero={'base':'zero','hidden_sizes':[],'operations':[]}
    assert np.array_equal(forward(x,execute(zero,task,state)),np.zeros((len(x),4)))
    scores=forward(x,state['best'])
    print('Starting metrics:',json.dumps(clustering_metrics(x,scores.argmax(axis=1)),indent=2))
    print('Mean scores:',scores.mean(axis=0))
    print('Score ranges:',scores.min(axis=0),scores.max(axis=0))
    print('Assigned raw means:')
    print(raw.groupby(scores.argmax(axis=1)).mean().round(3).to_string())
PY
python /workspace/bench.py