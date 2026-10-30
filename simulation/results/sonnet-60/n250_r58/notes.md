# Notes
Data: 250 rows, one covariate x0 in [0,1]. Mean is non-monotone in x0 (dips, then rises) and the spread grows strongly with x0.
Candidates: polynomial mean (deg 2-5) x log-sd polynomial (deg 1-3), Normal vs Student-t. Validation: 10-fold CV, 3 repeats (Normal) / 3 repeats, mean NLL on held-out folds.
Results (CV NLL): log-sd linear ~0.73-0.77; log-sd quadratic ~0.62-0.68; mean cubic+quad log-sd 0.6236; deg 4/5 mean or cubic log-sd did not improve meaningfully (0.618-0.627). Student-t gave no gain (df -> large).
Final: Normal, cubic mean, log-sd quadratic (7 fitted constants, fit on all data).
CV NLL of this configuration is 0.624; the selection among a small set of candidates adds slight optimism.
expected_nll: 0.64
