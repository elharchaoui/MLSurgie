
mkdir -p /workspace; cat > /workspace/load.py <<'EOF'
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
d=pd.read_csv('/inputs/data/support.csv')
E=pd.read_csv('/inputs/data/support_encoded.csv')
EOF
cat > /tmp/a.py <<'EOF'
exec(open('/workspace/load.py').read())
f=d[d.label==1]
print(f.describe().T)
for c,b in [('amount',[0,3,5,8,11,15,100,180,300,500,1000,3000]),('account_age_days',[-1,7,14,30,60,88,150,400,10000]),('txn_last_24h',[-1,0,1,2,3,4,5,6,8,20]),('distance_from_home_km',[0,34,60,100,200,500,5000])]:
    print(d.groupby(pd.cut(d[c],b)).label.agg(['mean','sum','count']))
d['night']=d.hour.isin([0,1,2,3,4,5])
print(d.groupby('night').label.agg(['mean','sum','count']))
print(pd.crosstab(f.merchant_category,f.card_present))
print(pd.crosstab(d.merchant_category,d.card_present))
EOF
python /tmp/a.py