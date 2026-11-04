from bench import *

# One mechanism: a centered additive metabolic-risk score.
# Stratified high/low contrasts are summarized in contrast_summary.csv.
# BMI conditional slope 32.4 contains shared HDL/TG risk: use 28.
# TG conditional slope 35.1 contains shared HDL/BMI/BP risk: use 30.
# BP conditional slope 13.9 is little altered by remaining risk: use 14.
# HDL conditional slope -13.0 is attenuated by sex imbalance: use -15.
# Sex contrast -8.4 is confounded by higher BMI/BP and lower HDL in group 2;
# their risk offsets account for roughly 18 units: use group2 minus group1 = -26.
# Other cholesterol measures duplicate this evidence and are not used.
coeffs={'bmi:standardized':28., 's5:standardized':30.,
        'bp:standardized':14., 's3:standardized':-15., "sex='2.0'":-26.}
sex2_fraction=float(ENC["sex='2.0'"].mean())
bias=float(Y.mean()+26*sex2_fraction)
rationale=(
    'Centered additive metabolic-risk circuit. Support mean is the baseline. '
    'Within strata of the other major risk factors, upper-minus-lower half mean '
    'contrasts give target-unit slopes per input SD of BMI 32.4, log-triglycerides '
    '35.1, BP 13.9, and HDL -13.0. BMI and log-TG contrast groups still differ in '
    'HDL and other risk markers, so shared evidence is discounted to 28 and 30; '
    'BP is rounded to 14. HDL is adjusted to -15 because the high-HDL group is '
    'enriched for sex 1. The stratified sex-2-minus-sex-1 target contrast -8.4 '
    'still contains higher BMI, BP and lower HDL risk, motivating -26 after '
    'subtracting those positive risks. Sex is centered at its support frequency; '
    'numeric inputs are already centered. Age, glucose, and redundant cholesterol '
    'measures are omitted to avoid noisy duplicate evidence. Weights are rounded '
    'from descriptive contrasts and nuisance imbalances, not optimized.'
)
program=linear_program(coeffs,bias,rationale)
write(program)
write(program,'/workspace/candidate_v1.json')
pred=forward(program)
print('Chosen coefficients',coeffs,'intercept',bias)
print('Candidate support metrics:',metrics(pred))
print('Prediction quantiles:',np.quantile(pred,[0,.05,.25,.5,.75,.95,1]))
print('Target mean',Y.mean(),'sex2 fraction',sex2_fraction)
for f in ['bmi','s5','bp','s3','sex']:
    print('\nResidual by',f, '\n',bins(f,target=Y-pred).round(2).to_string())
check=pd.DataFrame({'prediction':pred,'residual':Y-pred,'target':Y})
check['bin']=pd.qcut(pred,5)
print('\nCalibration\n',check.groupby('bin',observed=True).agg(n=('target','size'),prediction=('prediction','mean'),target=('target','mean'),residual=('residual','mean')).round(2).to_string())
