# Notes
Data: 250 rows, 1 covariate in [0,1]. Mean of y is nonlinear (dip near x=0.25, rise after), spread grows strongly with x.
Validation: 10-fold CV repeated 3x (parameters refit in each fold), mean NLL.
Candidates (Normal / StudentT / SkewNormal; polynomial mean deg 2-5, log-sd poly deg 1-3):
- Normal, cubic mean, quadratic log-sd: 0.631 (best; t gave identical, df large; skew-normal worse 0.648+)
- quartic/quintic mean or cubic log-sd: 0.633-0.637; linear log-sd: 0.78+
Final: Normal(cubic(x), exp(quadratic(x))), 7 constants. Residuals ~symmetric, slightly light-tailed.
expected_nll: 0.64
