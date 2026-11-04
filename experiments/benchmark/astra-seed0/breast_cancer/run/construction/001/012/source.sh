cat > /workspace/construct_v1.py <<'PY'
"""Six physical evidence channels with hand-assigned redundancy discounts.
Only descriptive moments of the construction split determine normalization and
one-dimensional Gaussian log-odds; no numerical parameter fitting is performed.
"""
from bench import *

# Normalized channel weights are explicit mechanism decisions, not searched.
# Size and indentation share r~0.5 and get about 2/3 of independent evidence.
# Surface and asymmetry share r~0.47 and are discounted; their negative
# within-class relation to size still gives them complementary information.
channel_weights={'log_size':1.8, 'indentation':1.6, 'texture':0.9,
                 'surface':0.75, 'asymmetry':0.4, 'relative_radius_error':1.05}

# Label-free piecewise-linear approximations of log physical measurements.
# Linear extrapolation is bounded outside the very broad physical range.
log_specs={'worst_radius':np.geomspace(6,42,19),
           'mean_radius':np.geomspace(5,35,17),
           'radius_error':np.geomspace(.075,3.5,25)}
W=[]; B=[]; encodings={}

def add_relu(feature,knot):
    w=np.zeros(len(names)); w[idx(feature)]=pre[feature]['scale']
    W.append(w); B.append(pre[feature]['mean']-knot)
    return len(W)-1

def log_spline(feature,knots):
    slopes=np.diff(np.log(knots))/np.diff(knots)
    jumps=np.r_[slopes[0], np.diff(slopes),-slopes[-1]]
    units=[add_relu(feature,t) for t in knots]
    return float(np.log(knots[0])), dict(zip(units,jumps))

for feature,knots in log_specs.items():
    encodings[feature]=log_spline(feature,knots)

# Positive and negative standardized inputs are exact linear pass-throughs.
def linear(feature,scale=1.0):
    w=np.zeros(len(names)); w[idx(feature)]=1
    k=len(W); W.extend([w,-w]); B.extend([0.,0.])
    return pre[feature]['mean']*scale, {k:pre[feature]['scale']*scale,k+1:-pre[feature]['scale']*scale}

def combine(items):
    offset=0.; coeff={}
    for factor,(b,terms) in items:
        offset+=factor*b
        for k,v in terms.items(): coeff[k]=coeff.get(k,0.)+factor*v
    return offset,coeff

channels={
    'log_size':encodings['worst_radius'],
    'indentation':linear('worst_concave_points'),
    'texture':combine([(0.5,linear('mean_texture',1/3.8)),(0.5,linear('worst_texture',1/5.4))]),
    'surface':linear('worst_smoothness'),
    'asymmetry':linear('worst_symmetry'),
    'relative_radius_error':combine([(1,encodings['radius_error']),(-1,encodings['mean_radius'])])}
W=np.asarray(W); B=np.asarray(B)
H=np.maximum(X@W.T+B,0)
output=np.zeros(len(W)); offset=0.; ledger={}
for name,(b,terms) in channels.items():
    v=b+sum(H[:,k]*a for k,a in terms.items())
    vb=v[build]; yb=y[build]
    m0=vb[yb==0].mean(); m1=vb[yb==1].mean()
    pooled_sd=np.sqrt(((vb[yb==0]-m0)**2).sum()+((vb[yb==1]-m1)**2).sum())/np.sqrt(len(build)-2)
    a=channel_weights[name]/pooled_sd
    center=(m0+m1)/2
    offset+=a*(b-center)
    for k,val in terms.items(): output[k]+=a*val
    ledger[name]={'mean_benign':m0,'mean_malignant':m1,'pooled_sd':pooled_sd,'standardized_evidence':channel_weights[name]}

score=H@output+offset
sb=score[build]; yb=y[build]
m0=sb[yb==0].mean(); m1=sb[yb==1].mean()
v0=sb[yb==0].var(ddof=1); v1=sb[yb==1].var(ddof=1)
var=((yb==0).sum()-1)*v0+((yb==1).sum()-1)*v1
var/=len(build)-2
slope=(m1-m0)/var
prior=np.log((yb.sum()+.5)/(len(yb)-yb.sum()+.5))
final_offset=prior+slope*(offset-(m0+m1)/2)
output*=slope
rationale=(
    'Six physical evidence channels distinguish malignant nuclei: log(worst radius), '
    'worst concave points, averaged mean/worst texture, worst smoothness, worst symmetry, '
    'and log(radius standard error / mean radius). Log transforms represent relative '
    'growth and heterogeneity and reduce right-skew; fixed logarithmic secants implement '
    'them with ReLUs. Each channel is centered at its two class means and divided by the '
    'pooled within-class standard deviation computed on a fixed 256-row construction '
    'partition. Explicit channel weights [1.8,1.6,0.9,0.75,0.4,1.05] discount strongly '
    'correlated size/indentation and smoothness/symmetry evidence while keeping texture '
    'and size-normalized variability independent. A one-dimensional Gaussian density '
    'ratio of the resulting scalar score provides an analytic slope and midpoint bias, '
    'plus smoothed class prior. No weight search or optimization. Local 85-row holdout '
    'is excluded from every supervised statistic. Missing indicators are unused because '
    'all support measurements are present.')
p=program([W,output[None,:]],[B,np.array([final_offset])],['relu'],rationale)
write(p)
write(p,ROOT/'candidate_v1.json')
with open(ROOT/'ledger_v1.json','w') as f:
    json.dump({'channels':ledger,'projection':{'mean0':m0,'mean1':m1,'var0':v0,'var1':v1,'pooled_variance':var,'slope':slope,'prior':prior}},f,indent=2)
if __name__=='__main__':
    print('Channel statistics',json.dumps(ledger,indent=2))
    print('Projected class moments',m0,m1,v0,v1,'slope',slope,'prior',prior)
    z=report(p)
    print(calibration(z).round(4).to_string())
    np.savez(ROOT/'predictions_v1.npz',score=score,logit=z)
PY
python /workspace/construct_v1.py