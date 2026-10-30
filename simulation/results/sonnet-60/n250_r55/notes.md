# Notes
Data: 250 rows, x0 in [0,1]. Mean is nonlinear (dip near 0.2, rises, plateaus); spread grows strongly with x0.
Validation: 10-fold CV, 3 repeats, mean held-out NLL. Candidates: polynomial mean degree 2-6, log-sd polynomial degree 1-3,
families Normal / StudentT / SkewNormal.
Results (CV NLL): cubic mean + quad log-sd Normal 0.596 (best simple); StudentT identical (df -> large); quad mean 0.69;
deg 4-6 means slightly worse (0.597-0.607); linear log-sd 0.70; skew normal 0.61; cubic log-sd 0.594 (negligible gain, not used).
Final: Normal(cubic(x0), exp(quadratic(x0))), 7 fitted constants, fit on all data (train NLL 0.580).
expected_nll: 0.60
