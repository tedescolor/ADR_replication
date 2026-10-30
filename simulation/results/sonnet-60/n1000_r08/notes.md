# Notes
Data: 1 covariate x0 in (0,1); mean is a nonlinear (cubic/quartic-like) function of x0, spread grows strongly with x0, and skewness changes with x0.
Validation: 10-fold CV (fixed seed), mean held-out NLL, models fit by max likelihood (BFGS, several skew starts).
Tried: Normal/StudentT with polynomial or Fourier mean and polynomial log-sd (best CV ~0.666; Student-t no gain);
SkewNormal with polynomial loc / log-scale / shape (degrees m,s,a): (3,2,0) 0.668, (3,2,2) 0.629, (5,2,2) 0.621, (4,2,2) 0.613 (best), (6,2,2) 0.626, (4,2,1) 0.653.
Final: SkewNormal, loc quartic, log-scale quadratic, shape quadratic in x0 (11 constants). Full-data train NLL 0.607.
CV is slightly optimistic from selecting among ~20 configs; the CV NLL of the chosen one is 0.613, so I add a small margin.
expected_nll: 0.63
