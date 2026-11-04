cat > /workspace/construct_v2.py <<'PY'
"""Heteroscedastic Gaussian evidence compiled into piecewise-quadratic ReLU units."""
import numpy as np
from bench import *
from construct_v1 import EVIDENCE, TEMPERATURE


def quadratic_program(a,b,c,rationale):
    """Compile sum_j a[k,j]*x_j^2 + b[k,j]*x_j + c[k] into ReLUs.

    x^2 is linearly interpolated on half-unit knots from -4.5 to +4.5,
    with linear tails. This is a fixed numeric approximation, not label fitting.
    """
    knots=np.arange(-4.,4.01,.5)
    initial_slope=-8.5
    initial_intercept=-18.
    units=[]; biases=[]; outputs=[]
    outbias=c+initial_intercept*a.sum(axis=1)
    for j,ix in enumerate(NUMERIC_INDICES):
        slope=b[:,j]+initial_slope*a[:,j]
        for sign in [1.,-1.]:
            w=np.zeros(X.shape[1]); w[ix]=sign
            units.append(w); biases.append(0.); outputs.append(sign*slope)
        for t in knots:
            w=np.zeros(X.shape[1]); w[ix]=1.
            units.append(w); biases.append(-t); outputs.append(a[:,j])
    return {'base':'zero','hidden_sizes':[len(units)],'activations':['relu'],
            'rationale':rationale,'operations':[
               {'op':'set','tensor':'layers.0.weight','value':np.array(units).tolist()},
               {'op':'set','tensor':'layers.0.bias','value':np.array(biases).tolist()},
               {'op':'set','tensor':'layers.1.weight','value':np.array(outputs).T.tolist()},
               {'op':'set','tensor':'layers.1.bias','value':outbias.tolist()}]}


def construct(rows):
    z=X[rows][:,NUMERIC_INDICES]; y=Y[rows]
    counts=np.bincount(y,minlength=3)
    means=np.stack([z[y==k].mean(axis=0) for k in range(3)])
    var=np.stack([z[y==k].var(axis=0,ddof=1) for k in range(3)])
    pooled=np.sum(var*(counts[:,None]-1),axis=0)/(len(rows)-3)
    # Same global shrinkage and temperature as v1. New mechanism: class-specific
    # spreads, partially pooled to prevent noisy small-class variance estimates.
    spread=.9*(.7*var+.3*pooled[None,:])+.1
    evidence=EVIDENCE[None,:]/TEMPERATURE
    precision=evidence/spread
    a=-.5*precision
    b=means*precision
    priors=(counts+1)/(counts.sum()+3)
    c=np.log(priors)-.5*np.sum(means**2*precision+evidence*np.log(spread),axis=1)
    rationale=(
       'Revision of the pooled Gaussian prototype mechanism: allow each cultivar its own '
       'chemical spread. Support shows cultivar 2 has broader phenolic, ash, hue, and magnesium '
       'distributions, cultivar 1 has broader proline, and cultivar 3 has broader color intensity. '
       'Shared variance incorrectly treats rare high-magnesium/high-phenolic cultivar 2 wines '
       'as ever-stronger evidence for cultivar 1, matching the development error pattern. '
       'Class-conditional variances are 70% empirical variance and 30% pooled within-class '
       'variance, followed by 10% shrinkage toward unit standardized variance. Class means, '
       'prior counts, 1.25 temperature, and phenolic redundancy weights match candidate 1. '
       'The Gaussian log likelihood is expanded into per-coordinate quadratic, linear, and '
       'constant terms. A 247-unit ReLU hidden layer represents 13 independent squares via '
       'half-unit linear interpolation on [-4.5,4.5] (linear tails), plus signed identity units. '
       'This approximates the quadratic likelihood without training. Missing indicators are '
       'unused. All parameters come only from support counts, means and variances.'
    )
    return quadratic_program(a,b,c,rationale),means,spread

if __name__=='__main__':
    train,val=holdout_indices()
    local,_,_=construct(train)
    print('Fixed honest holdout:',metrics(local,X[val],Y[val]))
    program,means,spread=construct(np.arange(len(Y)))
    print('Full support:',metrics(program))
    print('Worst rows:',worst_rows(program,8).round(4).to_string(index=False))
    save(program); save(program,'/workspace/candidate_v2.json')
PY
python /workspace/construct_v2.py