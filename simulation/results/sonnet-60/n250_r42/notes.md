# Notes
Data: 250 rows, 1 covariate x0 in [0,1]. Mean of y is nonlinear in x0 (dip near 0.25, rise to ~0.8) and the spread grows strongly for x0>0.7.
Validation: 5-fold CV repeated 3-4 times (different shuffles), MLE refit per fold, mean held-out NLL.
Candidates (CV NLL): Normal with polynomial mean (deg 2-5) and polynomial log-sd (deg 1-3); also StudentT and SkewNormal variants.
Best: Normal, cubic mean, quadratic log-sd: 0.711-0.713. Linear log-sd ~0.81 (clearly worse); t/skew-normal gave no gain (residual skew/kurtosis ~0.2/0).
Final: Normal(mu(x), exp(quadratic(x))), 7 constants, fit by MLE on all data.
The CV estimate has mild optimism from selecting among ~18 configs; I add a small margin.
expected_nll: 0.73
