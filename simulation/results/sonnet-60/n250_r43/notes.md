# Notes
Data: 1 covariate x0 in [0,1], y nonlinear in x0 (dip near 0.25, rises after) with spread growing strongly in x0.
Candidates: Normal / StudentT / SkewNormal with polynomial mean (deg 2-5) and polynomial log-sd (deg 1-3), fit by ML.
Validation: 10-fold CV repeated 3x (random splits), mean held-out NLL.
Results (CV NLL): Normal cubic mean + quadratic log-sd 0.622 (best); t and skew-normal no gain (0.622-0.63); higher degrees worse; linear log-sd 0.747.
Final: Normal(cubic(x0), exp(quadratic(x0))), 7 constants, fit on all 250 rows (train NLL 0.593).
CV NLL 0.622 is slightly optimistic due to model selection among ~36 candidates; add a small margin.

expected_nll: 0.64
