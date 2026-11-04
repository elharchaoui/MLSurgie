"""Simple NumPy executor and descriptive clustering diagnostics; no fitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

TASK_PATH = Path('/inputs/task.json')

def load_data():
    task = json.loads(TASK_PATH.read_text())
    raw = pd.read_csv('/inputs/data/support.csv')
    encoded = pd.read_csv('/inputs/data/support_encoded.csv')
    X = encoded[task['encoded_features']].to_numpy(dtype=np.float64)
    return task, raw, X

def materialize(program, input_size, output_size):
    if program['base'] != 'zero':
        raise ValueError('This bench only materializes explicit zero-base programs.')
    sizes = [input_size] + program['hidden_sizes'] + [output_size]
    tensors = {}
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        tensors[f'layers.{i}.weight'] = np.zeros((n_out,n_in),dtype=np.float32)
        tensors[f'layers.{i}.bias'] = np.zeros(n_out,dtype=np.float32)
    for op in program['operations']:
        tensor = tensors[op['tensor']]
        if op['op'] == 'fill':
            tensor.fill(op['value'])
        elif op['op'] == 'set':
            values = np.asarray(op['value'],dtype=np.float32)
            if values.shape != tensor.shape:
                raise ValueError((op['tensor'],values.shape,tensor.shape))
            tensor[:] = values
        elif op['op'] in ('set_entries','add_entries'):
            for entry in op['entries']:
                idx = tuple(entry['index'])
                if op['op'] == 'set_entries':
                    tensor[idx] = entry['value']
                else:
                    tensor[idx] += entry['value']
        else:
            raise ValueError(op['op'])
    return tensors

def forward(X, program, output_size=4):
    tensors = materialize(program, X.shape[1],output_size)
    hidden = program['hidden_sizes']
    activations = program.get('activations',['relu']*len(hidden))
    value = np.asarray(X,dtype=np.float32)
    for i in range(len(hidden)+1):
        value = value @ tensors[f'layers.{i}.weight'].T + tensors[f'layers.{i}.bias']
        if i < len(hidden):
            activation = activations[i]
            if activation == 'relu':
                value = np.maximum(value,0)
            elif activation == 'tanh':
                value = np.tanh(value)
            elif activation == 'sigmoid':
                value = 1/(1+np.exp(-np.clip(value,-80,80)))
            elif activation != 'identity':
                raise ValueError(activation)
    return value

def silhouette(X, assignment):
    """Standard Euclidean silhouette, computed exactly on supplied rows."""
    assignment = np.asarray(assignment)
    unique, counts = np.unique(assignment,return_counts=True)
    n = len(X)
    if len(unique)<2 or len(unique)>=n:
        return {'silhouette':-1.0, 'cluster_sizes':dict(zip(unique.tolist(),counts.tolist()))}
    distance = cdist(X,X,metric='euclidean')
    means = np.column_stack([distance[:,assignment==group].mean(axis=1) for group in unique])
    samples = np.zeros(n)
    for col, group in enumerate(unique):
        mask = assignment==group
        a = means[mask,col]*counts[col]/max(counts[col]-1,1)
        b = np.min(means[mask][:,np.arange(len(unique))!=col],axis=1)
        denom = np.maximum(a,b)
        samples[mask] = np.divide(b-a,denom,out=np.zeros_like(a),where=denom>0)
        if counts[col]==1:
            samples[mask]=0
    return {
        'silhouette':float(samples.mean()),
        'cluster_sizes':dict(zip(unique.tolist(),counts.tolist())),
        'per_cluster_silhouette':{int(group):float(samples[assignment==group].mean()) for group in unique},
        'negative_silhouette_rows':int((samples<0).sum()),
        'silhouette_quantiles':np.quantile(samples,[0,.1,.25,.5,.75,.9,1]).tolist(),
    }

def write_program(path, weights, biases, rationale, activations=None):
    hidden = [len(bias) for bias in biases[:-1]]
    program = {'base':'zero', 'hidden_sizes':hidden,
               'activations':activations if activations is not None else ['relu']*len(hidden),
               'rationale':rationale, 'operations':[]}
    for i,(weight,bias) in enumerate(zip(weights,biases)):
        program['operations'].append({'op':'set','tensor':f'layers.{i}.weight','value':np.asarray(weight).tolist()})
        program['operations'].append({'op':'set','tensor':f'layers.{i}.bias','value':np.asarray(bias).tolist()})
    Path(path).write_text(json.dumps(program,indent=2)+'\n')
    return program

if __name__=='__main__':
    # Trivial zero network verifies dimensions and head semantics without fitting.
    task, raw, X = load_data()
    zero = {'base':'zero','hidden_sizes':[], 'operations':[]}
    assert np.array_equal(forward(X[:3],zero,task['output_size']),np.zeros((3,4)))
    print('Bench zero-network check passed:', X.shape)
