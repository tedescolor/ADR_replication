# Notes
Data: 1 covariate, 250 rows. Mean is nonlinear (dip then rise), spread grows strongly with x0.
Candidates: Normal / StudentT, polynomial mean degree 2-6, log-sd polynomial degree 1-3, fit by MLE (BFGS).
Validation: 10-fold CV repeated over 3 random splits, mean held-out NLL.
Results (CV NLL): mean cubic + log-sd quadratic Normal 0.665 (best); StudentT no gain; log-sd linear ~0.79; mean deg2 0.734; higher degrees slightly worse.
Final: Normal(cubic(x0), exp(quadratic(x0))), 7 constants, refit on all data (in-sample NLL 0.639).
expected_nll: 0.67
