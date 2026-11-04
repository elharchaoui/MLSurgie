
cat > /tmp/b.py <<'EOF'
exec(open('/workspace/load.py').read())
d['velo']=d.txn_last_24h>=7
d['far']=d.distance_from_home_km>200
d['new']=d.account_age_days<=30
d['big']=d.amount>300
d['small']=d.amount<11
d['night']=d.hour<=5
d['cnp']=d.card_present=='no'
d['risky']=d.merchant_category.isin(['electronics','gaming','gift_cards'])
d['cm']=d.country_mismatch=='yes'
flags=['velo','far','new','big','small','night','cnp','risky','cm']
for a in flags:
    print(a, d[a].sum(), d[d[a]].label.sum(), round(d[d[a]].label.mean(),3))
f=d[d.label==1]
print(f[flags].astype(int).value_counts().head(40))
r=d[~d.velo & ~d.far]
print('rest',len(r),r.label.sum())
import itertools
for a,b in itertools.combinations(flags[2:],2):
    s=r[r[a]&r[b]]
    print(a,b,len(s),s.label.sum(),round(s.label.mean(),3))
EOF
python /tmp/b.py