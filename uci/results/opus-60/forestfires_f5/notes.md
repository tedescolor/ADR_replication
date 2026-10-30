# Notes

Data: 416 rows, 12 jittered covariates, y heavy-tailed (max 1092). About a third of y lie in a
noisy spike around 0 (100 values are negative, down to -1.37): looks like a point mass at zero
blurred by noise, plus a positive heavy-tailed part.

Validation: repeated 5-fold CV (2-3 repeats, fixed seeds), mean held-out NLL on the raw y scale.

Marginal families (CV NLL):
- Normal spike + LogNormal: **2.836** (train 2.820)
- Laplace spike + LogNormal: 2.849; StudentT spike + LogNormal: 2.836 (df -> inf, = Normal)
- Normal + Gamma: 2.972
- Normal + 2 LogNormal: 2.847; 2 Normal + LogNormal: 2.845; T + 2 LN: 2.851 (overfit)

Covariates: binned summaries of P(y>0.8) and mean log y versus every covariate show no clear
pattern. Adding each covariate linearly (standardized) to the logit of the spike weight or to
the LogNormal location changed CV NLL by between -0.004 and +0.016, i.e. noise. No covariate
was kept.

Final: Mixture(0.336 * Normal(-0.262, 0.489) + 0.664 * LogNormal(mu=1.180, sigma=1.806)),
6 literal constants (5 free parameters), no covariate dependence. Chosen because it has the lowest CV NLL and is the simplest.
The heavy tail makes the test-set mean noisy; estimate is CV plus a small allowance.

expected_nll: 2.84
