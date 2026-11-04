import sys
sys.path.insert(0,'/workspace')
from bench import *
f=raw.iloc[build].copy()
# Physically motivated channels; no labels are involved in the transforms.
f['log_size']=np.log(f.worst_radius)
f['indentation']=f.worst_concave_points
f['texture']=0.5*(f.mean_texture/3.8+f.worst_texture/5.4)
f['surface']=f.worst_smoothness
f['asymmetry']=f.worst_symmetry
f['relative_radius_error']=np.log(f.radius_error/f.mean_radius)
channels=['log_size','indentation','texture','surface','asymmetry','relative_radius_error']
means=f.groupby('label')[channels].mean()
r=f[channels]-f.groupby('label')[channels].transform('mean')
sd=np.sqrt((r*r).sum()/(len(f)-2))
print('Channel means, pooled sd, standardized class separation:')
print(pd.DataFrame({'benign':means.loc[0],'malignant':means.loc[1],'sd':sd,'separation':(means.loc[1]-means.loc[0])/sd}).round(4).to_string())
print('\nWithin-class correlations:')
print(r.corr().round(2).to_string())
print('\nClass quantiles for relative radius error:')
print(f.groupby('label').relative_radius_error.quantile([.1,.25,.5,.75,.9]).unstack().round(3).to_string())