
cat > /workspace/build.py <<'EOF'
exec(open('/workspace/bench.py').read())
import sys
nin=len(F)
# layer-1 sigmoid step detectors: (name, list of (feature, weight), bias)
def numstep(feat, t, raw_slope, direction):  # direction +1: x>t ; -1: x<t
    k=raw_slope*sc(feat)*direction
    return ([(feat+':standardized',k)], -k*std(feat,t))
S={}
S['new']=numstep('account_age_days',35,0.8,-1)
S['cnp']=([("card_present='no'",10)],-5)
S['risky']=([("merchant_category='electronics'",10),("merchant_category='gaming'",10),("merchant_category='gift_cards'",10)],-5)
S['big']=numstep('amount',180,0.3,+1)
S['velo']=numstep('txn_last_24h',6.5,4,+1)
S['small']=numstep('amount',11,2,-1)
S['night']=numstep('hour',5.5,4,-1)
S['far']=numstep('distance_from_home_km',250,0.05,+1)
names=list(S)
W0=np.zeros((len(names),nin)); b0=np.zeros(len(names))
for i,n in enumerate(names):
    for f,w in S[n][0]: W0[i,FI[f]]=w
    b0[i]=S[n][1]
# layer-2 ANDs
base=-5.4
U=[('A3',['new','cnp','risky','big'],2,2.7),('A4',['new','cnp','risky','big'],3,3.0),
   ('B3',['velo','small','cnp'],2,8.4),
   ('C3',['night','far','cnp'],2,8.4)]
W1=np.zeros((len(U),len(names))); b1=np.zeros(len(U)); W2=np.zeros((1,len(U)))
for j,(n,cs,th,w) in enumerate(U):
    for c in cs: W1[j,names.index(c)]=1
    b1[j]=-th; W2[0,j]=w
r=lambda a: np.round(a,5).tolist()
cand={"base":"zero","hidden_sizes":[len(names),len(U)],"activations":["sigmoid","relu"],
 "rationale":"L1 sigmoid soft-step detectors: "+", ".join(names)+". L2 ReLU AND units: "+"; ".join(f"{n}=relu(sum{cs}-{th}) w={w}" for n,cs,th,w in U)+f". Output bias {base}=log-odds of residual fraud rate (~0.45%) outside the three signatures (new account+CNP+risky merchant+big amount; card-testing burst: >=7 txns/24h + tiny amount + CNP; night + >250km from home + CNP). Weights = log-odds of each signature cell rate (~95%, shrunk) minus base.",
 "operations":[{"op":"set","tensor":"layers.0.weight","value":r(W0)},{"op":"set","tensor":"layers.0.bias","value":r(b0)},
 {"op":"set","tensor":"layers.1.weight","value":r(W1)},{"op":"set","tensor":"layers.1.bias","value":r(b1)},
 {"op":"set","tensor":"layers.2.weight","value":r(W2)},{"op":"fill","tensor":"layers.2.bias","value":base}]}
json.dump(cand,open('/workspace/candidate.json','w'))
E=pd.read_csv('/inputs/data/support_encoded.csv'); X=E[F].values; y=E.label.values
z,h=forward(cand,X)
print('AP',ap(y,z),'LL',logloss(y,z),'meanp',np.mean(1/(1+np.exp(-z))),y.mean())
# calibration by score bins
p=1/(1+np.exp(-z)); bins=pd.cut(p,[0,0.005,0.01,0.03,0.1,0.3,0.6,0.9,1])
print(pd.DataFrame({'p':p,'y':y}).groupby(bins).agg(['mean','count']))
np.save('/tmp/z.npy',z)
EOF
python /workspace/build.py