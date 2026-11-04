
cat > /tmp/c.py <<'EOF'
exec(open('/workspace/load.py').read())
pd.set_option('display.width',250); pd.set_option('display.max_rows',200); pd.set_option('display.max_columns',30)
risky=d.merchant_category.isin(['electronics','gaming','gift_cards']); cnp=d.card_present=='no'
A=(d.account_age_days<=30)&cnp&risky&(d.amount>200)
B=(d.txn_last_24h>=6)&cnp&risky&(d.amount<12)
C=(d.hour<=5)&(d.distance_from_home_km>250)&cnp
for n,m in [('A',A),('B',B),('C',C)]: print(n,m.sum(),d[m].label.sum())
rest=d[~(A|B|C)]
print(len(rest),rest.label.sum())
print(rest[rest.label==1][['amount','hour','account_age_days','txn_last_24h','merchant_category','card_present','country_mismatch','distance_from_home_km']])
vel=d[cnp&risky]
print(pd.crosstab(vel.txn_last_24h,[vel.label, vel.amount<12]))
EOF
python /tmp/c.py