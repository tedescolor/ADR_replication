# Notes
Data: 250 rows, 1 covariate. Mean of y is nonlinear in x0 (dip near 0.27, rises after), spread grows strongly for x0>0.6.
Candidates: Normal / StudentT, polynomial mean (deg 3-6), polynomial log-sigma (deg 1-3), fitted by ML.
Validation: 3x repeated 5-fold CV on mean NLL. Best: Normal, cubic mean, quadratic log-sigma, CV NLL 0.591
(StudentT best 0.5945 with deg 4; higher degrees overfit; linear log-sigma much worse, ~0.73).
Final: Normal with cubic mean and quadratic log-sigma fitted on all data (7 parameters, in-sample NLL 0.574).
Expected out-of-sample NLL is about the CV value.

expected_nll: 0.592
