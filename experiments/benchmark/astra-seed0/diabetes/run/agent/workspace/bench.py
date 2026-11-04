import json
from pathlib import Path
import numpy as np
import pandas as pd

TASK=json.loads(Path('/inputs/task.json').read_text())
FEATURES=TASK['encoded_features']
RAW=pd.read_csv('/inputs/data/support.csv')
ENC=pd.read_csv('/inputs/data/support_encoded.csv')
X=ENC[FEATURES].to_numpy(dtype=float)
Y=ENC['target'].to_numpy(dtype=float)

def forward(program, x=X):
    widths=[x.shape[1]]+program['hidden_sizes']+[TASK['output_size']]
    tensors={}
    for i,(a,b) in enumerate(zip(widths,widths[1:])):
        tensors[f'layers.{i}.weight']=np.zeros((b,a),dtype=float)
        tensors[f'layers.{i}.bias']=np.zeros(b,dtype=float)
    for op in program['operations']:
        arr=tensors[op['tensor']]
        if op['op']=='fill': arr.fill(op['value'])
        elif op['op']=='set': arr[...]=np.asarray(op['value'])
        elif op['op'] in ['set_entries','add_entries']:
            for e in op['entries']:
                if op['op']=='set_entries': arr[tuple(e['index'])]=e['value']
                else: arr[tuple(e['index'])]+=e['value']
        else: raise ValueError(op['op'])
    h=x.copy()
    acts=program.get('activations',['relu']*len(program['hidden_sizes']))
    for i in range(len(widths)-1):
        h=h@tensors[f'layers.{i}.weight'].T+tensors[f'layers.{i}.bias']
        if i<len(acts):
            if acts[i]=='relu': h=np.maximum(0,h)
            elif acts[i]=='tanh': h=np.tanh(h)
            elif acts[i]=='sigmoid': h=1/(1+np.exp(-np.clip(h,-700,700)))
            elif acts[i]!='identity': raise ValueError(acts[i])
    return h[:,0] if h.shape[1]==1 else h

def metrics(pred, y=Y):
    err=y-np.asarray(pred)
    return {'rmse':float(np.sqrt(np.mean(err**2))),
            'mae':float(np.mean(np.abs(err))),
            'r2':float(1-np.mean(err**2)/np.var(y)),
            'bias_prediction_minus_target':float(-np.mean(err))}

def linear_program(coeffs,bias,rationale):
    w=np.zeros(len(FEATURES))
    for f,v in coeffs.items(): w[FEATURES.index(f)]=v
    return {'base':'zero','hidden_sizes':[], 'activations':[], 'rationale':rationale,
            'operations':[{'op':'set','tensor':'layers.0.weight','value':[w.tolist()]},
                          {'op':'set','tensor':'layers.0.bias','value':[float(bias)]}]}

def write(program, path='/workspace/candidate.json'):
    Path(path).write_text(json.dumps(program,indent=2))

def bins(feature, target=Y, q=4, data=RAW):
    df=pd.DataFrame({'x':data[feature], 'target':target})
    if df.x.nunique()>q: df['bin']=pd.qcut(df.x,q,duplicates='drop')
    else: df['bin']=df.x
    return df.groupby('bin',observed=True).agg(n=('target','size'),xmin=('x','min'),xmax=('x','max'),xmean=('x','mean'),ymean=('target','mean'),ystd=('target','std'))

if __name__=='__main__':
    p=linear_program({},float(Y.mean()),'Support mean baseline for forward-pass verification.')
    assert np.allclose(forward(p),Y.mean())
    print('Forward baseline verified:',metrics(forward(p)))
