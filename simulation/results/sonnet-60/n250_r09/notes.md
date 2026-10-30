# Notes
Data: 250 rows, one covariate x0 in [0,1]. Mean is non-monotone in x0 (dips near 0.25, rises after 0.5); spread is heteroscedastic (smallest around x0=0.5, large at x0 near 1).
Validation: 10-fold CV, 2 repeats, mean held-out NLL. Compared Normal / StudentT / SkewNormal with polynomial mean (deg 2-5) and polynomial log-sigma (deg 1-3).
Best: Normal, cubic mean, quadratic log-sigma, CV NLL ~0.691-0.695 (t and skew-normal no better; deg 2 mean/linear sigma much worse at 0.84-0.90). Final fit on all data (train NLL 0.671). 7 constants.
expected_nll: 0.70
