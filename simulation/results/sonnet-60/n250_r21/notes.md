# Notes
Data: 1 covariate x0 in (0,1), y with nonlinear mean (dip then rise) and sd growing with x0.
Validation: 10-fold CV (single split seed 0), Normal/StudentT/SkewNormal with polynomial mean (deg 2-6)
and polynomial log-sd (deg 1-3), small L2 penalty. Normal with cubic mean + quadratic log-sd was best
(CV NLL 0.711); t and skew-normal gave no gain; deg 2 mean was much worse (0.780).
Final: Normal(cubic(x0), exp(quadratic(x0))), 7 fitted constants, fit on all data (train NLL 0.689).
Expected NLL is CV-based.

expected_nll: 0.72
