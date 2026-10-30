# Notes
Data: 250 rows, 1 covariate. Mean of y is nonlinear in x0 (dip then rise), sd grows with x0.
Validation: 10-fold CV x3 repeats, comparing Normal/StudentT with polynomial mean (deg 2-5) and polynomial log-sd (deg 1-2).
Best: Normal, cubic mean, quadratic log-sd (CV NLL 0.708). Student-t and higher degrees were not better; linear log-sd clearly worse (0.81).
Final: Normal(cubic(x0), exp(quadratic(x0))), 7 fitted constants, fit on all data (train NLL 0.689).
Expected NLL ≈ CV estimate.
expected_nll: 0.71
