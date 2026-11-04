import json, numpy as np, pandas as pd
T=json.load(open('/inputs/task.json'))
P={p['name']:p for p in T['preprocessing']}
F=T['encoded_features']; FI={f:i for i,f in enumerate(F)}
def std(name,t): return (t-P[name]['mean'])/P[name]['scale']
def sc(name): return P[name]['scale']
def forward(c,X):
    W={}
    sizes=[X.shape[1]]+c['hidden_sizes']+[1]
    for i in range(len(sizes)-1):
        W[f'layers.{i}.weight']=np.zeros((sizes[i+1],sizes[i])); W[f'layers.{i}.bias']=np.zeros(sizes[i+1])
    for op in c['operations']:
        t=W[op['tensor']]
        if op['op']=='fill': t[...]=op['value']
        elif op['op']=='set': t[...]=np.array(op['value'])
        else:
            for e in op['entries']:
                if op['op']=='set_entries': t[tuple(e['index'])]=e['value']
                else: t[tuple(e['index'])]+=e['value']
    h=X
    acts=c.get('activations',['relu']*len(c['hidden_sizes']))
    for i in range(len(c['hidden_sizes'])):
        h=h@W[f'layers.{i}.weight'].T+W[f'layers.{i}.bias']
        a=acts[i]
        h=np.maximum(h,0) if a=='relu' else np.tanh(h) if a=='tanh' else 1/(1+np.exp(-h)) if a=='sigmoid' else h
    L=len(c['hidden_sizes'])
    return h@W[f'layers.{L}.weight'].T[:,0]+W[f'layers.{L}.bias'][0], h
def ap(y,s):
    o=np.argsort(-s,kind='mergesort'); y=y[o]; tp=np.cumsum(y); k=np.arange(1,len(y)+1)
    return (tp/k)[y==1].sum()/y.sum()
def logloss(y,z):
    p=1/(1+np.exp(-z)); p=np.clip(p,1e-7,1-1e-7); return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
