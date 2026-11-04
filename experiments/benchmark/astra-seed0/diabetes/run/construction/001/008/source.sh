cat > /workspace/profile.py <<'PY'
from bench import *

Z=pd.DataFrame({f.split(':')[0]:X[:,i] for i,f in enumerate(FEATURES) if f.endswith(':standardized')})
Z['sex']=RAW.sex-1
Z['target']=Y

def contrast(f, controls, target=Y):
    d=Z.copy()
    d['target']=target
    c=[]
    for g,q in controls:
        k=g+'_bin';c.append(k)
        d[k]=pd.qcut(d[g],q,labels=False,duplicates='drop') if g!='sex' else d[g]
    rows=[]
    for key,cell in d.groupby(c,observed=True):
        if len(cell)<8:continue
        threshold=.5 if f=='sex' else cell[f].median()
        low=cell[cell[f]<=threshold]
        high=cell[cell[f]>threshold]
        if len(low)<3 or len(high)<3:continue
        row={'strata':str(key),'n':len(cell),'nlo':len(low),'nhi':len(high),'dx':high[f].mean()-low[f].mean(),'dy':high.target.mean()-low.target.mean()}
        for g in Z.columns:
            if g not in [f,'target']:row['d_'+g]=high[g].mean()-low[g].mean()
        row['w']=len(low)*len(high)/len(cell)
        rows.append(row)
    out=pd.DataFrame(rows)
    summary={k:np.average(out[k],weights=out.w) for k in out.columns if k not in ['strata','n','nlo','nhi','w']}
    summary['n']=out.n.sum()
    summary['slope']=summary['dy']/summary['dx']
    return out,summary

if __name__=='__main__':
    print('Baseline:',metrics(np.full(len(Y),Y.mean())))
    rows=[]
    for f,controls in [('bmi',[('s5',3),('bp',2),('sex',2)]),('s5',[('bmi',3),('bp',2),('sex',2)]),('bp',[('bmi',3),('s5',3)]),('s3',[('bmi',3),('s5',3)]),('sex',[('bmi',3),('s5',3)])]:
        t,s=contrast(f,controls)
        t.to_csv('/workspace/contrast_'+f+'.csv',index=False)
        s['feature']=f
        rows.append(s)
    summary=pd.DataFrame(rows).set_index('feature')
    print(summary.round(3).to_string())
    summary.to_csv('/workspace/contrast_summary.csv')
PY
PYTHONPATH=/workspace python /workspace/bench.py
PYTHONPATH=/workspace python /workspace/profile.py