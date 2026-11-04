"""Minimal analytic repair of two inconsistent centroid-head entries."""
import json
from pathlib import Path
import numpy as np
from bench import load, execute, forward, clustering_metrics, write_candidate

task,state,raw,x=load()
p=state['best']['parameters']
w=np.asarray(p['layers.0.weight'],dtype=float)
b=np.asarray(p['layers.0.bias'],dtype=float)
# The frequent visitor prototype is already in the observed frequent/low-basket
# density mode. A nearest-centroid score must have negative squared-norm bias,
# not an unconditional positive bonus. Compute it directly, without fitting.
frequent_bias=-float(np.sum(np.square(w[1]/2)))
# For the high-spend/infrequent prototype, the current basket coordinate is
# -2.214 standard deviations, or -100.5 USD: outside the entire data range.
# Its sign is inconsistent with that segment's large-basket density mode.
large_basket_weight=abs(float(w[3,4]))
program={
 'base':'best','base_hash':state['best']['hash'],
 'hidden_sizes':state['best']['hidden_sizes'],'activations':state['best']['activations'],
 'rationale':(
   'Repair two inconsistent entries in the existing nearest-centroid head, leaving all other coefficients unchanged. '
   'Cluster 1 has the frequent-visitor/low-basket prototype, but its +2.90794 intercept creates a large unconditional bonus '
   'and swallows 615/960 support rows. Set that intercept to -||W_1/2||^2 = '+str(frequent_bias)+', '
   'the exact analytic intercept for its unchanged prototype. '
   'Cluster 3 combines high annual spend, very few visits, and long recency with an impossible negative basket prototype '
   '(W_3,4/2=-2.214 standardized, about -100.5 USD, versus observed minimum 9.61 USD). '
   'Reverse only this standardized basket coefficient to +4.42806482315, restoring a roughly 235.6 USD large-basket prototype. '
   'The descriptive basket/visit histogram has 131 of 142 baskets above 140 USD at <=2 visits/month, confirming the sign. '
   'Keep the existing small-spend and online-dominant centroids, all remaining biases, and zero age weights. '
   'No optimization or iterative centroid updates; only sign diagnosis and the nearest-centroid norm identity.'
 ),
 'operations':[
  {'op':'set_entries','tensor':'layers.0.bias','entries':[{'index':[1],'value':frequent_bias}]},
  {'op':'set_entries','tensor':'layers.0.weight','entries':[{'index':[3,4],'value':large_basket_weight}]}
 ]
}
network=execute(program,task,state)
scores=forward(x,network)
a=scores.argmax(axis=1)
metrics=clustering_metrics(x,a)
print('Repair metrics:',json.dumps(metrics,indent=2))
print('Repaired raw means:')
print(raw.groupby(a).mean().round(3).to_string())
# Deterministic subset audit; neither thresholds nor prototypes are estimated
# from any labels or from this audit's metric.
holdout=np.arange(len(x))%4==0
print('Subset check:',json.dumps(clustering_metrics(x[holdout],a[holdout]),indent=2))
print('Changed bias:',b[1],'=>',frequent_bias,'basket:',w[3,4],'=>',large_basket_weight)
write_candidate(program)
Path('/workspace/iteration1_support_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
