import sys
sys.path.insert(0,'/workspace')
from bench import *
f=raw.iloc[build]
for name in ['worst_radius','worst_concave_points','worst_texture','worst_smoothness','radius_error']:
 print('\n',name)
 bins=pd.qcut(f[name],8,duplicates='drop')
 t=f.groupby(bins,observed=True).agg(n=('label','size'),pos=('label','sum'),mean=(name,'mean'),lo=(name,'min'),hi=(name,'max'))
 t['log_odds']=np.log((t.pos+0.5)/(t.n-t.pos+0.5))
 print(t.round(3).to_string(index=False))
print('\nWithin class standard errors:')
print(f.groupby('label')[['mean_radius','mean_texture','radius_error','perimeter_error','area_error']].agg(['mean','std']).round(3).to_string())