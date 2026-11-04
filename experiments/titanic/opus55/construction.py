
import json,math
t=json.load(open('/inputs/task.json'))
F=t['encoded_features'];ix={f:i for i,f in enumerate(F)}
H=14;W=[[0.0]*len(F) for _ in range(H)];b=[0.0]*H
def s(h,f,v):W[h][ix[f]]=v
P1,P2,P3,FE,MA,AG,SB,PA,FA,EC="pclass='1'","pclass='2'","pclass='3'","sex='female'","sex='male'","age:standardized","sibsp:standardized","parch:standardized","fare:standardized","embarked='C'"
s(0,FE,1);s(0,P1,1);s(0,P2,1);b[0]=-1
s(1,FE,1);s(1,P3,1);b[1]=-1
s(2,MA,1);s(2,P1,1);b[2]=-1
s(3,MA,1);s(3,P2,1);b[3]=-1
s(4,MA,1);s(4,P3,1);b[4]=-1
s(5,AG,-3);s(5,FE,-10);b[5]=-3.3
s(6,AG,-3);s(6,FE,-10);b[6]=-4.3
s(7,AG,1)
s(8,AG,-1)
s(9,SB,1.22);s(9,PA,0.797);b[9]=-2.039
s(10,SB,1.22);s(10,PA,0.797);b[10]=0.961
s(11,SB,1.22);s(11,PA,0.797);b[11]=-0.039
s(12,FA,1)
s(13,EC,1)
out=[2.8,0.0,-0.5,-1.7,-1.9,2.0,-2.0,-0.3,0.2,-0.9,0.25,-0.25,0.2,0.25]
ob=0.0
def pred(x):
    h=[max(0,sum(W[j][i]*x[i] for i in range(len(x)))+b[j]) for j in range(H)]
    z=sum(o*v for o,v in zip(out,h))+ob
    return 1/(1+math.exp(-z))
ll=0;acc=0
for e in t['support_examples']:
    p=pred(e['encoded']);y=e['label']
    ll+=-math.log(p if y else 1-p);acc+=(p>0.5)==y
n=len(t['support_examples']);print(ll/n,acc/n)
cand={"base":"zero","hidden_sizes":[H],"rationale":"Hidden units encode sex x class cells (female 1/2, female 3, male 1/2/3), a clipped boy-age ramp, age ReLU pair, clipped family-size presence and large-family (>3) penalty, fare ReLU, embarked C. Output weights are prior logits from Titanic survival rates per cell.",
"operations":[{"op":"set","tensor":"layers.0.weight","value":W},{"op":"set","tensor":"layers.0.bias","value":b},{"op":"set","tensor":"layers.1.weight","value":[out]},{"op":"fill","tensor":"layers.1.bias","value":ob}]}
json.dump(cand,open('/work/candidate.json','w'))
