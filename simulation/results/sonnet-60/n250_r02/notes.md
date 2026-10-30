# Notes
Data: 250 rows, one covariate x0 in [0,1]. Mean of y is non-monotone in x0 (dip near 0.2, rising to ~1 at x0=1); the spread is heteroscedastic (large near 0 and near 1, small in the middle).

Candidates: Normal / StudentT / SkewNormal with a polynomial mean (order 3-7) and polynomial log-sd (order 1-4), fitted by max likelihood (BFGS).
Validation: 10-fold CV, mean held-out NLL, 1 repetition (a 2-repetition run was aborted for time).
- Constant log-sd or linear log-sd: 0.77. Quadratic log-sd: 0.574-0.576, so the variance structure matters most.
- Mean order 3/4/5, log-sd order 2/3/4: all about 0.574 (differences are noise).
- StudentT 0.576 (no gain), SkewNormal 0.616 (worse), mean order 6-7 slightly worse.

Final: Normal, cubic mean, quadratic log-sd, 7 fitted constants, about 20 in total. In-sample NLL 0.552; CV NLL 0.574.
Caveat: the model selection used the same CV folds, so the estimate is slightly optimistic.

expected_nll: 0.58
