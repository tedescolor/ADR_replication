# Notes
Data: 250 rows, 1 covariate. Mean of y is nonlinear in x0 (dip near 0.2, rise to ~1 at x0=1); spread is small mid-range and large at both ends (heteroscedastic).
Validation: 10-fold CV (single random split, seed 0), mean held-out NLL. Compared Normal vs StudentT, polynomial mean degree 3-7, log-sd polynomial degree 1-3.
Result: log-sd degree 1 is bad (~0.79); degree 2 is best (~0.59). Mean degree 4 was best (0.589). StudentT gave no gain (0.590). Higher degrees overfit.
Final: Normal(quartic poly mean, exp(quadratic) sd) in t=x0-0.5, fit by ML on all data. 8 constants plus 2 in the centering/offsets.
CV-selection over a grid is slightly optimistic, so I add a small margin.
expected_nll: 0.60
