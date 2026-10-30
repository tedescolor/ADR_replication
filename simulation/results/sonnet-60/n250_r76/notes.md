# Notes
Data: 1 covariate, mean is nonlinear in x0 (dip near 0.2, rise after), spread grows with x0.
Candidates: Normal / StudentT, polynomial mean (deg 2-6), polynomial log-sd (deg 1-3).
Validation: repeated 5-fold CV (3-6 repeats), mean held-out NLL.
Results: Normal cubic mean + quadratic log-sd best, CV NLL ~0.671 (deg2 mean 0.77, linear log-sd 0.80, StudentT no gain).
Final: Normal(cubic(x), exp(quadratic(x))), 7 constants, fitted by ML on all data (train NLL 0.645).
expected_nll: 0.675
