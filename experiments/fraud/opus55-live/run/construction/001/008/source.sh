
cat > /tmp/c.py <<'EOF'
exec(open('/workspace/load.py').read())
pd.set_option('display.width',250); pd.set_option('display.max_rows',200); pd.set_option('display.max_columns',20)
v=d[d.txn_last_24h>=6]
print(v.groupby(['label']).agg({'amount':['min','max'],'txn_last_24h':['min','max']}))
print(pd.crosstab([v.label],[v.card_present,v.merchant_category]))
print(v[v.label==0].sort_values('amount').head(15))
far=d[d.distance_from_home_km>250]
print(far[far.label==0]); print(pd.crosstab(far.label,[far.country_mismatch,far.card_present]))
print(far[far.label==1].merchant_category.value_counts())
n=d[(d.account_age_days<=40)]
print(n.groupby('label').amount.describe())
print(pd.crosstab([n.label],[n.card_present,n.merchant_category]))
n2=n[n.amount>150]; print(pd.crosstab(n2.label,[n2.card_present, n2.merchant_category.isin(['electronics','gaming','gift_cards'])]))
print(n2.sort_values('amount')[['amount','account_age_days','merchant_category','card_present','label']])
EOF
python /tmp/c.py