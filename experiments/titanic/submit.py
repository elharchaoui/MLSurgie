"""Materialize a Codex-designed circuit; only unit conversion, no data fitting."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
iteration=int(sys.argv[1])
design=json.loads(Path(sys.argv[2]).read_text())
folder=ROOT/'runs/titanic-codex-001/iterations'/f'{iteration:03d}'
context=json.loads((folder/'request.json').read_text())
names=context['encoded_features']
pre={x['name']:x for x in context['preprocessing']}
weights=[]; biases=[]; outputs=[]

def unit(encoded=None, raw=None, bias=0., out=0.):
    w=[0.]*len(names)
    for name,value in (encoded or {}).items(): w[names.index(name)]=value
    for name,value in (raw or {}).items():
        w[names.index(name+':standardized')]=value*pre[name]['scale']
        bias += value*pre[name]['mean']
    weights.append(w);biases.append(bias);outputs.append(out)

for sex in ['female','male']:
    for cls in ['1','2','3']:
        unit({f'sex={sex!r}':1,f'pclass={cls!r}':1},bias=-1,out=design['group_logits'][sex+cls])
# Family benefit saturates at one relative; large-family cost saturates after four excess relatives.
family={'sibsp':1,'parch':1}
unit(raw=family,out=design['family_bonus'])
unit(raw=family,bias=-1,out=-design['family_bonus'])
unit(raw=family,bias=-3,out=-design['large_family_penalty'])
unit(raw=family,bias=-7,out=design['large_family_penalty'])
# Bounded male-child ramp: 0 at age>=14, 1 at age<=4; female gating suppresses it.
unit({"sex='male'":4},raw={'age':-.1},bias=1.4-4,out=design['male_child_bonus'])
unit({"sex='male'":4},raw={'age':-.1},bias=.4-4,out=-design['male_child_bonus'])
# Female child term can differ from adult baseline, also clipped.
unit({"sex='female'":4},raw={'age':-.1},bias=1.4-4,out=design['female_child_bonus'])
unit({"sex='female'":4},raw={'age':-.1},bias=.4-4,out=-design['female_child_bonus'])
unit({'age:missing':1},out=design['missing_age'])
unit({"embarked='C'":1},out=design['port_c'])
unit({"embarked='Q'":1,"sex='male'":1},bias=-1,out=design['male_port_q'])
# Older-age penalty begins at 40 and caps at 70.
unit(raw={'age':.05},bias=-2,out=-design['older_age_penalty'])
unit(raw={'age':.05},bias=-3.5,out=design['older_age_penalty'])
# Low fare effect only among third-class passengers; bounded between fare 5 and 15.
unit({"pclass='3'":4},raw={'fare':-.1},bias=1.5-4,out=-design['low_fare_penalty'])
unit({"pclass='3'":4},raw={'fare':-.1},bias=.5-4,out=design['low_fare_penalty'])
# Female third-class family penalty, activates for at least two relatives, capped at four.
unit({"sex='female'":20,"pclass='3'":20},raw=family,bias=-2-40,out=-design.get('female3_family_penalty',0))
unit({"sex='female'":20,"pclass='3'":20},raw=family,bias=-4-40,out=design.get('female3_family_penalty',0))
# Male second-class child bonus distinct from third-class.
unit({"sex='male'":4,"pclass='2'":4},raw={'age':-.1},bias=1.4-8,out=design.get('male2_child_extra',0))
unit({"sex='male'":4,"pclass='2'":4},raw={'age':-.1},bias=.4-8,out=-design.get('male2_child_extra',0))
# Zero-fare indicator approximated by a ramp over [0, 1].
unit(raw={'fare':-1},bias=1,out=-design.get('zero_fare_penalty',0))
unit(raw={'fare':-1},bias=0,out=design.get('zero_fare_penalty',0))
while len(weights)<32: unit()
assert len(weights)==32
program={'base':'zero','hidden_sizes':[32],'rationale':design['rationale'],'operations':[
    {'op':'set','tensor':'layers.0.weight','value':weights},
    {'op':'set','tensor':'layers.0.bias','value':biases},
    {'op':'set','tensor':'layers.1.weight','value':[outputs]},
    {'op':'set','tensor':'layers.1.bias','value':[0.]},
]}
(folder/'design.json').write_text(json.dumps(design,indent=2)+'\n')
temp=folder/'response.tmp'
temp.write_text(json.dumps(program,indent=2)+'\n')
temp.rename(folder/'response.json')
print('Submitted direct weights for iteration',iteration)
