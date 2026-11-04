
cat > /tmp/c.py <<'EOF'
exec(open('/workspace/load.py').read())
pd.set_option('display.width',250); pd.set_option('display.max_rows',200); pd.set_option('display.max_columns',20)
v=d[d.txn_last_24h>=5]
print(pd.crosstab([v.label,v.txn_last_24h],[v.card_present,v.merchant_category]))
print(v[v.label==1].amount.describe())
vv=v[(v.card_present=='no')&v.merchant_category.isin(['electronics','gaming','gift_cards'])]
print(vv[['amount','hour','account_age_days','txn_last_24h','merchant_category','country_mismatch','distance_from_home_km','label']])
far=d[d.distance_from_home_km>200]
print(far[['amount','hour','account_age_days','txn_last_24h','merchant_category','card_present','country_mismatch','distance_from_home_km','label']].sort_values('hour'))
EOF
python /tmp/c.py