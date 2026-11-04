"""Local network IR forward pass and summary metrics (no parameter fitting)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

TASK = json.loads(Path('/inputs/task.json').read_text())
RAW = pd.read_csv('/inputs/data/support.csv')
ENC = pd.read_csv('/inputs/data/support_encoded.csv')
FEATURES = TASK['encoded_features']
X = ENC[FEATURES].to_numpy(dtype=float)
Y = ENC['label'].to_numpy(dtype=int)
NUMERIC_INDICES = np.array([j for j,c in enumerate(FEATURES) if c.endswith(':standardized')])
NUMERIC_NAMES = [FEATURES[j].split(':')[0] for j in NUMERIC_INDICES]

def tensors(program):
    assert program['base'] == 'zero'
    widths = [X.shape[1]] + program['hidden_sizes'] + [TASK['output_size']]
    p = {}
    for i,(ni,no) in enumerate(zip(widths[:-1],widths[1:])):
        p[f'layers.{i}.weight'] = np.zeros((no,ni),dtype=float)
        p[f'layers.{i}.bias'] = np.zeros(no,dtype=float)
    for op in program['operations']:
        a = p[op['tensor']]
        if op['op'] == 'set': a[:] = np.asarray(op['value'])
        elif op['op'] == 'fill': a.fill(op['value'])
        else:
            for e in op['entries']:
                index=tuple(e['index'])
                if op['op'] == 'set_entries': a[index]=e['value']
                elif op['op'] == 'add_entries': a[index]+=e['value']
                else: raise ValueError(op)
    return p

def forward(program, x):
    p=tensors(program)
    a=np.asarray(x,dtype=float)
    acts=program.get('activations', ['relu']*len(program['hidden_sizes']))
    for i in range(len(program['hidden_sizes'])+1):
        a=a @ p[f'layers.{i}.weight'].T + p[f'layers.{i}.bias']
        if i<len(acts):
            if acts[i]=='relu': a=np.maximum(0,a)
            elif acts[i]=='tanh': a=np.tanh(a)
            elif acts[i]=='sigmoid': a=1/(1+np.exp(-np.clip(a,-700,700)))
            elif acts[i]!='identity': raise ValueError(acts[i])
    return a

def probabilities(logits):
    z=logits-np.max(logits,axis=1,keepdims=True)
    p=np.exp(z)
    return p/np.sum(p,axis=1,keepdims=True)

def metrics(program,x=X,y=Y):
    p=probabilities(forward(program,x))
    pred=p.argmax(axis=1)
    truep=p[np.arange(len(y)),y]
    cm=np.zeros((3,3),dtype=int)
    np.add.at(cm,(y,pred),1)
    return {'rows':int(len(y)),'log_loss':float(np.mean(-np.log(np.clip(truep,1e-15,1)))),
            'accuracy':float(np.mean(pred==y)),'confusion':cm.tolist(),
            'worst_true_prob':float(truep.min())}

def worst_rows(program,n=10):
    p=probabilities(forward(program,X))
    truep=p[np.arange(len(Y)),Y]
    ix=np.argsort(truep)[:n]
    d=RAW.iloc[ix].copy()
    d['row']=ix
    d['prediction']=p[ix].argmax(axis=1)
    d['true_probability']=truep[ix]
    return d

def save(program,path='/workspace/candidate.json'):
    a=tensors(program)
    assert all(np.isfinite(v).all() for v in a.values())
    assert max(np.abs(v).max() for v in a.values())<=TASK['limits']['max_abs_weight']
    assert sum(v.size for v in a.values())<=TASK['limits']['max_parameters']
    Path(path).write_text(json.dumps(program,indent=2,allow_nan=False))

def linear_program(weights,bias,rationale):
    return {'base':'zero','hidden_sizes':[],'activations':[], 'rationale':rationale,
            'operations':[{'op':'set','tensor':'layers.0.weight','value':np.asarray(weights).tolist()},
                          {'op':'set','tensor':'layers.0.bias','value':np.asarray(bias).tolist()}]}

def holdout_indices():
    rng=np.random.default_rng(42)
    val=[]
    for k in range(3):
        ix=np.where(Y==k)[0]
        rng.shuffle(ix)
        val.extend(ix[:max(1,int(round(.2*len(ix))))])
    val=np.array(sorted(val))
    train=np.setdiff1d(np.arange(len(Y)),val)
    return train,val

if __name__=='__main__':
    p=linear_program(np.zeros((3,X.shape[1])),np.log([.2,.3,.5]),'Forward-pass test: constant priors')
    np.testing.assert_allclose(probabilities(forward(p,X))[:1],[[.2,.3,.5]])
    print('Network forward pass verified on constant multiclass prior.')
