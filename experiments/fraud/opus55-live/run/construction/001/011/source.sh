
cat > /tmp/c.py <<'EOF'
exec(open('/workspace/load.py').read())
pd.set_option('display.width',250); pd.set_option('display.max_rows',200); pd.set_option('display.max_columns',30)
risky=d.merchant_category.isin(['electronics','gaming','gift_cards']); cnp=d.card_present=='no'
x=d[cnp&risky&(d.account_age_days<=60)]
print(pd.crosstab(pd.cut(x.account_age_days,[-1,20,25,30,35,40,50,60]),[x.label,pd.cut(x.amount,[0,150,200,250,300,5000])]))
x=d[(d.txn_last_24h>=5)]
print(pd.crosstab([x.txn_last_24h],[x.label,x.card_present,pd.cut(x.amount,[0,10,12,15,30,5000])]))
x=d[(d.distance_from_home_km>150)]
print(pd.crosstab([pd.cut(x.distance_from_home_km,[150,200,250,300,400,5000])],[x.label,x.card_present,x.hour<=5]))
rest=d[~(((d.account_age_days<=30)&cnp&risky&(d.amount>200))|((d.txn_last_24h>=7)&(d.amount<12))|((d.hour<=5)&(d.distance_from_home_km>250)&cnp))]
for c in ['merchant_category','card_present','country_mismatch']:
    print(rest.groupby(c).label.agg(['mean','sum','count']))
for c in ['amount','hour','account_age_days','txn_last_24h']:
    print(rest.groupby(pd.qcut(rest[c],5,duplicates='drop')).label.agg(['mean','sum','count']))
EOF
python /tmp/c.py