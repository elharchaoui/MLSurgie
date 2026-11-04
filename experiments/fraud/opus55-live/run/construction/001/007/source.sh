
cat > /tmp/c.py <<'EOF'
exec(open('/workspace/load.py').read())
pd.set_option('display.width',200); pd.set_option('display.max_rows',200)
print(d[d.txn_last_24h>=6].sort_values('txn_last_24h'))
print(d[d.distance_from_home_km>150].sort_values('distance_from_home_km'))
EOF
python /tmp/c.py