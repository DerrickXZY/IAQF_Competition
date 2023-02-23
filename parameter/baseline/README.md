# Parameters for Baseline Model
Here are the explanations for the output:

## VECM Parameters
`k_ar_diff`: number of lags in VECM. The reason why it is called `k_ar_diff` is that `k_ar` stands for number of lags in the VAR representation. `k_ar_diff` = `k_ar` - 1.

`alpha`: coefficients of error correction terms(s)

`beta`: coefficients to build error correction terms

`gamma`: coefficients of lag terms

`det_coef`: coefficients of deterministic terms, i.e. constant and trend term here.

See more parameter explanations at https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.vecm.VECMResults.html. For VECM model representation, see https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.vecm.VECM.html


## VAR Parameters
`k_ar`: number of lags

`params`: all parameters

See more parameter explanations at https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.var_model.VARResults.html.