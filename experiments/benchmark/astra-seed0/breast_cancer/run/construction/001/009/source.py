import sys
sys.path.insert(0,'/workspace')
from bench import *
f=raw.iloc[build]
print('Build label counts:',f.label.value_counts().to_dict())
print('Standardized class mean, sd, and pooled standardized mean gap')
stats=[]
for name in raw.columns[:-1]:
 a=f.loc[f.label==0,name];b=f.loc[f.label==1,name]
 v=((len(a)-1)*a.var()+(len(b)-1)*b.var())/(len(a)+len(b)-2)
 stats.append([name,a.mean(),b.mean(),a.std(),b.std(),(b.mean()-a.mean())/np.sqrt(v)])
print(pd.DataFrame(stats,columns=['feature','benign_mean','malignant_mean','benign_sd','malignant_sd','separation']).round(4).to_string(index=False))
select=['worst_radius','worst_concave_points','mean_concave_points','mean_texture','worst_texture','worst_smoothness','worst_symmetry','radius_error','worst_concavity','worst_compactness','worst_fractal_dimension']
z=f[select]-f.groupby('label')[select].transform('mean')
print('\nWithin class correlation\n',z.corr().round(2).to_string())