cat > /workspace/bench.py <<'PY'
"""Local network runner, strictly descriptive metrics and declarative program writer."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
ROOT=Path('/workspace')
task=json.load(open('/inputs/task.json'))
raw=pd.read_csv('/inputs/data/support.csv')
encoded=pd.read_csv('/inputs/data/support_encoded.csv')
names=task['encoded_features']
X=encoded[names].to_numpy(dtype=float)
y=encoded.label.to_numpy(dtype=int)
pre={f['name']:f for f in task['preprocessing']}
# Fixed label-independent split; local holdout is never used for statistics.
rng=np.random.default_rng(821)
order=rng.permutation(len(y))
check=np.sort(order[:85]); build=np.sort(order[85:])

def idx(feature): return names.index(feature+':standardized')
def threshold(feature,t): return (t-pre[feature]['mean'])/pre[feature]['scale']

def program(weights,biases,activations=None,rationale=''):
    widths=[np.asarray(w).shape[0] for w in weights[:-1]]
    operations=[]
    for i,(w,b) in enumerate(zip(weights,biases)):
        operations += [{'op':'set','tensor':f'layers.{i}.weight','value':np.asarray(w).tolist()},
                       {'op':'set','tensor':f'layers.{i}.bias','value':np.asarray(b).reshape(-1).tolist()}]
    return dict(base='zero',hidden_sizes=widths,activations=activations or ['relu']*len(widths),rationale=rationale,operations=operations)

def tensors(p):
    widths=[X.shape[1]]+p['hidden_sizes']+[task['output_size']]
    ts={}
    for i in range(len(widths)-1):
        ts[f'layers.{i}.weight']=np.zeros((widths[i+1],widths[i]))
        ts[f'layers.{i}.bias']=np.zeros(widths[i+1])
    for op in p['operations']:
        a=ts[op['tensor']]
        if op['op']=='fill': a[:]=op['value']
        elif op['op']=='set': a[:]=op['value']
        else:
            for entry in op['entries']:
                pos=tuple(entry['index'])
                if op['op']=='set_entries': a[pos]=entry['value']
                elif op['op']=='add_entries': a[pos]+=entry['value']
                else: raise ValueError(op)
    return ts

def forward(p,x=X):
    ts=tensors(p); h=x
    for i in range(len(p['hidden_sizes'])+1):
        h=h@ts[f'layers.{i}.weight'].T+ts[f'layers.{i}.bias']
        if i<len(p['hidden_sizes']):
            act=p.get('activations',['relu']*len(p['hidden_sizes']))[i]
            if act=='relu': h=np.maximum(h,0)
            elif act=='tanh': h=np.tanh(h)
            elif act=='sigmoid': h=expit(h)
            elif act!='identity': raise ValueError(act)
    return h[:,0]

def metrics(z,labels=y):
    labels=np.asarray(labels); z=np.asarray(z); prob=expit(z)
    pos=labels.sum(); neg=len(labels)-pos
    auc=(rankdata(z)[labels==1].sum()-pos*(pos+1)/2)/(pos*neg)
    return {'n':len(labels),'log_loss':float(np.mean(np.logaddexp(0,z)-labels*z)),
            'accuracy':float(np.mean((z>=0)==labels)),'auc':float(auc),
            'brier':float(np.mean((prob-labels)**2)), 'mean_p':float(prob.mean()),'rate':float(labels.mean())}

def report(p):
    z=forward(p)
    for name,indices in [('build',build),('holdout',check),('all',np.arange(len(y)))]:
        print(name,metrics(z[indices],y[indices]))
    return z

def calibration(z,indices=build):
    f=pd.DataFrame({'z':z[indices],'p':expit(z[indices]),'y':y[indices]})
    f['bin']=pd.qcut(f.p,10,duplicates='drop')
    return f.groupby('bin',observed=True).agg(n=('y','size'),p=('p','mean'),rate=('y','mean'),lo=('z','min'),hi=('z','max'))

def write(p,path=ROOT/'candidate.json'):
    path=Path(path)
    json.dump(p,open(path,'w'),indent=2,allow_nan=False)
    return path

if __name__=='__main__':
    p=program([np.zeros((1,len(names)))],[np.zeros(1)],rationale='bench zero test')
    assert np.all(forward(p)==0)
    assert abs(metrics(forward(p))['log_loss']-np.log(2))<1e-12
    print('Forward and loss baseline verified. split',len(build),len(check))
PY
python /workspace/bench.py