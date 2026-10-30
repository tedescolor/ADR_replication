Data: 1 covariate, n=250. Mean is nonlinear in x0 (dip near 0.2, then rising); spread grows strongly with x0.
Candidates (10-fold CV mean NLL, one random split): Normal/StudentT/SkewNormal; polynomial mean degree 2-5; log-scale polynomial degree 1-3.
Best: SkewNormal, quartic mean, quadratic log-scale, constant shape: CV NLL 0.508. Normal best 0.527; StudentT gave no gain; log-scale degree 3 overfit.
Final: refit on all data (9 parameters + 3.56 shape = 9 constants plus Horner coefficients). CV selection among ~20 configs is mildly optimistic, so I add a small margin.
expected_nll: 0.53
