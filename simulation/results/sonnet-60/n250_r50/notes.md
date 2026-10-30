# Notes
Data: 250 rows, one covariate x0 in [0,1]. Mean of y is nonlinear in x0 (dip near 0.2, rises after 0.5) and the spread grows strongly with x0.
Candidates: Normal / StudentT, polynomial mean (deg 2-6) and polynomial log-sigma (deg 1-3), fit by ML.
Validation: 10-fold CV repeated 3 times (different shuffles), mean held-out NLL.
Results (CV NLL): Normal cubic mean + quadratic log-sigma 0.594 (best); deg 4-6 means and cubic log-sigma worse; StudentT gave no gain (df -> large); quadratic mean 0.670; linear log-sigma ~0.73.
Final: Normal, cubic mean, log-sigma quadratic, 7 constants, fit on all data (train NLL 0.574).
expected_nll: 0.60
