"""Four density-defined retail prototypes, placed once from explicit regions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from bench import load_data, write_program, forward, silhouette

task, raw, X = load_data()
names = ['Budget regulars','Frequent loyal shoppers','Digital-first shoppers','Occasional premium shoppers']
# Independent histogram regions, not nearest-centroid assignments or iterative updates.
# A clear online-share gap lies at 0.50-0.60. Offline basket values have
# a trough near 100. The remaining annual-spend histogram has its trough near 2250.
region = np.where(raw.online_share.to_numpy()>0.55,2,
         np.where(raw.avg_basket.to_numpy()>100,3,
         np.where(raw.annual_spend.to_numpy()>2250,1,0)))
centroids = np.stack([X[region==k].mean(axis=0) for k in range(4)])
behavior = np.array([name.endswith(':standardized') and not name.startswith('age:')
                     for name in task['encoded_features']],dtype=bool)
centroids[:,~behavior]=0.0
# In behavioral standardized space, argmax(2 c dot x - c dot c)
# is exactly Euclidean nearest-prototype assignment. Age is uncorrelated noise.
W = 2*centroids
b = -np.sum(centroids**2,axis=1)
rationale = (
    'Four interpretable density-mode prototypes in the five standardized purchasing-behavior features. '
    'Support histograms reveal an online-share gap at 0.50-0.60, a large-basket trough near USD100, '
    'and a spend trough near USD2250 among low-online small-basket customers. '
    'Define disjoint seed regions once: online_share>0.55 -> digital-first; otherwise avg_basket>100 '
    '-> occasional premium; otherwise annual_spend>2250 -> frequent loyal; remainder -> budget regulars. '
    'Take each region mean once, without reassignment or iterative fitting. '
    'Output order is budget regulars, frequent loyal, digital-first, occasional premium. '
    'Raw purchasing-feature means respectively are approximately '
    '(1357,5.20,21.07,10.57,0.156), (4077,11.41,34.26,3.67,0.104), '
    '(2726,2.58,67.88,14.59,0.859), (5516,1.42,232.09,39.93,0.243). '
    'Age has near-zero correlation with all behavior and similar means in the four regions, so its weights '
    'are zero; missing indicators are also zero. Linear output row k is 2*c_k and bias is -sum(c_k^2), '
    'an exact nearest-prototype circuit with no hidden layers. Weights are derived only from explicit '
    'threshold-region means and geometry.'
)
program = write_program('/workspace/candidate.json',[W],[b],rationale)
score = forward(X,program,task['output_size'])
assignment = np.argmax(score,axis=1)
# Confirm the written layer reproduces the intended geometric rule.
distance2 = np.sum((X[:,None,behavior]-centroids[None,:,behavior])**2,axis=2)
assert np.array_equal(assignment,np.argmin(distance2,axis=1))
report = silhouette(X,assignment)
print('Support diagnostics:',json.dumps(report,indent=2))
print('Region-to-network contingency:\n',pd.crosstab(region,assignment).to_string())
print('Prototype standardized coordinates:\n',np.round(centroids[:,behavior],4))
print('Network group descriptions:\n',raw.assign(cluster=assignment).groupby('cluster').mean().round(3).to_string())
Path('/workspace/iteration1_diagnostics.json').write_text(json.dumps({
    'mechanism':'one-pass means of histogram-defined regions',
    'cluster_names':names,'seed_counts':np.bincount(region,minlength=4).tolist(),
    'centroids':centroids.tolist(),'support_report':report},indent=2)+'\n')
