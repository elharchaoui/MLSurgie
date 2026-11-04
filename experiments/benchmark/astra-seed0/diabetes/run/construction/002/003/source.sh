cat > /workspace/build_v2.py <<'PY'
from bench import *

base=json.loads(Path('/workspace/candidate_v1.json').read_text())
w=np.asarray(base['operations'][0]['value'][0],dtype=float)
b=float(base['operations'][1]['value'][0])
# Calibration geometry from broad support score bands: p~52 -> y~72,
# p~101 -> y~99. Round the implied low-risk slope (~0.54) to 0.5
# and join the unchanged additive model at p=100.
# This is a single new mechanism, not a change to the risk coefficients.
knot=100.
program={
 'base':'zero', 'hidden_sizes':[2], 'activations':['relu'],
 'rationale':base['rationale']+' New mechanism: attenuate the low-score extrapolation. '
     'The lowest score decile has mean prediction 51.9 but target 72.1, whereas '
     'the decile around prediction 101.1 has target 98.7. The implied low-risk '
     'response slope is about 0.54, rounded to 0.5, with the identity response '
     'retained above score 100. Two ReLUs encode p_new=100+ReLU(p-100)-0.5*ReLU(100-p). '
     'Thus the favorable-risk tail is shrunk toward a nonzero disease baseline, '
     'without altering moderate/high predictions or any feature effect.',
 'operations':[
     {'op':'set','tensor':'layers.0.weight','value':[w.tolist(),(-w).tolist()]},
     {'op':'set','tensor':'layers.0.bias','value':[b-knot,knot-b]},
     {'op':'set','tensor':'layers.1.weight','value':[[1.,-.5]]},
     {'op':'set','tensor':'layers.1.bias','value':[knot]}
 ]}
write(program)
write(program,'/workspace/candidate_v2.json')
p=forward(program);p0=forward(base)
assert np.allclose(p,p0+.5*np.maximum(0,100-p0))
print('Candidate support metrics:',metrics(p))
print('Rows changed:',int(np.sum(p0<100)))
print('Low-score target mean',Y[p0<100].mean(),'old mean',p0[p0<100].mean(),'new mean',p[p0<100].mean())
# Fixed disjoint slices diagnose stability; no selection over parameter values.
for fold in range(5):
 m=np.arange(len(Y))%5==fold
 print('Slice',fold,'n',int(m.sum()),'previous',round(metrics(p0[m],Y[m])['rmse'],3),'current',round(metrics(p[m],Y[m])['rmse'],3))
PY
PYTHONPATH=/workspace python /workspace/build_v2.py