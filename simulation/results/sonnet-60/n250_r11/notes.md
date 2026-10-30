# Notes
Data: 250 rows, one covariate x0 in (0,1). Mean of y is nonlinear in x0 (dip near 0.2, rise after);
spread grows strongly for x0>0.6 and is left-skewed there.
Validation: 3x repeated 5-fold CV on mean NLL, comparing Normal/StudentT with polynomial mean (deg 2-5) and
polynomial log-sd (deg 1-3). Best simple: Normal, cubic mean, quadratic log-sd (CV 0.611); Student-t gave no gain.
Final: SkewNormal(cubic mean, exp(quadratic) scale, shape linear in x0): CV NLL 0.595, 9 constants.
Params fitted on all data with a small ridge penalty. Training NLL is slightly lower; CV is the honest estimate.
expected_nll: 0.60
