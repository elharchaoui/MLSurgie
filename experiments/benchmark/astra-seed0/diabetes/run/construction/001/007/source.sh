PYTHONPATH=/workspace python - <<'PY'
from bench import *
print('Baseline:', metrics(np.full(len(Y),Y.mean())))
print('Correlations\n',RAW.corr().round(3).to_string())
# Slope contrasts read from upper and lower half sample means in strata of other major risks.
Z=pd.DataFrame({f.split(':')[0]:X[:,i] for i,f in enumerate(FEATURES) if f.endswith(':standardized')})
Z['sex']=RAW.sex-1
Z['target']=Y

def conditional_contrasts(f, controls):
    d=Z.copy()
    c=[]
    for g,q in controls:
        k=g+'_bin';c.append(k)
        d[k]=pd.qcut(d[g],q,labels=False,duplicates='drop') if g!='sex' else d[g]
    rows=[]
    for key,cell in d.groupby(c,observed=True):
        if len(cell)<8:continue
        low=cell[cell[f]<=cell[f].median()]
        high=cell[cell[f]>cell[f].median()]
        if len(low)<3 or len(high)<3:continue
        row={'strata':str(key),'n':len(cell),'nlo':len(low),'nhi':len(high),'dx':high[f].mean()-low[f].mean(),'dy':high.target.mean()-low.target.mean()}
        for g,_ in controls:row['d_'+g]=high[g].mean()-low[g].mean()
        row['w']=len(low)*len(high)/len(cell)
        rows.append(row)
    out=pd.DataFrame(rows)
    dx=np.average(out.dx,weights=out.w);dy=np.average(out.dy,weights=out.w)
    print('\n',f,'controlled by',controls,'n=',out.n.sum(),'weighted dx=',round(dx,3),'dy=',round(dy,3),'slope=',round(dy/dx,3))
    print('Remaining control imbalances:',{g:round(np.average(out['d_'+g],weights=out.w),3) for g,_ in controls})
    print(out.drop(columns=['w']).round(2).to_string(index=False))
for f,controls in [
    ('bmi',[('s5',3),('bp',2),('sex',2)]),
    ('s5',[('bmi',3),('bp',2),('sex',2)]),
    ('bp',[('bmi',3),('s5',3)]),
    ('s3',[('bmi',3),('s5',3)]),
    ('sex',[('bmi',3),('s5',3)]),
]:conditional_contrasts(f,controls)
PY