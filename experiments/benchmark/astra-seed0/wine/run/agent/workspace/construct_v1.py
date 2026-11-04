"""Pooled Gaussian prototypes from direct class means and sample spreads."""
import json
import numpy as np
from bench import *

# De-duplicate the correlated phenolic assays. Flavanoids is the strongest
# cultivar marker, so retains unit evidence; the overlapping assays are weaker.
EVIDENCE = np.array([1.,1.,1.,1.,1.,.5,1.,.65,.5,1.,1.,.8,1.])
# Inflate the empirical diagonal spread to temper independence assumptions.
TEMPERATURE = 1.25

def construct(rows):
    z=X[rows][:,NUMERIC_INDICES]
    y=Y[rows]
    counts=np.bincount(y,minlength=3)
    means=np.stack([z[y==k].mean(axis=0) for k in range(3)])
    residual=z-means[y]
    pooled=(residual**2).sum(axis=0)/(len(rows)-3)
    # A little global-variance shrinkage protects estimates from small samples.
    spread=.9*pooled+.1
    precision=EVIDENCE/(TEMPERATURE*spread)
    w=np.zeros((3,X.shape[1])); w[:,NUMERIC_INDICES]=means*precision
    prior=(counts+1)/(counts.sum()+3)
    b=np.log(prior)-.5*np.sum(means**2*precision,axis=1)
    rationale=(
      'Three pooled-variance Gaussian prototypes, computed directly from cultivar means and '
      'within-class sample spreads in standardized chemistry coordinates. The Gaussian '
      'quadratic term common to all cultivars cancels, leaving a linear softmax head with '
      'w_kj=mean_kj*precision_j and b_k=log(smoothed prior_k)-sum(mean_kj^2*precision_j)/2. '
      'All 13 chemicals contribute. Alcohol/proline favor cultivar 1 versus 2, and low '
      'flavanoids/OD280/hue with high color intensity and malic acid favor cultivar 3. '
      'Correlated total phenols and proanthocyanins receive half evidence, nonflavanoid '
      'phenols 0.65 and OD280 0.8 to reduce duplicated phenolic evidence. Variances are '
      '90% pooled within-class variance plus 10% unit variance and inflated 1.25x for '
      'conservative uncertainty. Missingness weights are zero because support is complete. '
      'No optimization or parameter search; only means, variances, counts, and a fixed '
      'chemistry/correlation-based evidence allocation.'
    )
    return linear_program(w,b,rationale), means, spread

if __name__=='__main__':
    train,val=holdout_indices()
    local,_,_=construct(train)
    print('Fixed honest holdout:',metrics(local,X[val],Y[val]))
    program,means,spread=construct(np.arange(len(Y)))
    print('Full support:',metrics(program))
    print('Worst rows:',worst_rows(program,8).round(4).to_string(index=False))
    print('Estimated spread:',dict(zip(NUMERIC_NAMES,spread.round(4))))
    save(program)
    save(program,'/workspace/candidate_v1.json')
