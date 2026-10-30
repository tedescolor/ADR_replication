# Notes
Data: 250 rows, 1 covariate. Mean is nonlinear in x0 (dips then rises), spread grows with x0 (min near 0.5).
Validation: 10-fold CV (shuffled, seed 0/1), mean test NLL; params fit by ML (BFGS).
Candidates (CV NLL): Normal cubic mean + linear log-sd 0.82; quad log-sd 0.700; cubic log-sd 0.701;
quartic/quintic mean ~0.702-0.705; Student-t no gain (0.700); skew-normal worse (0.706-0.73).
Final: Normal, cubic mean, log-sd quadratic in x0 (7 constants, fit on all data).
CV estimate 0.700; slight optimism from model selection among few candidates.
expected_nll: 0.71
