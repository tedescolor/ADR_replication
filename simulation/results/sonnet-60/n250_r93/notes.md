# Notes
Data: 250 rows, one covariate x0 in [0,1]. Mean of y is nonlinear in x0; spread is non-monotone (smallest near x0~0.45, large near 1).
Validation: 10-fold CV (shuffled, seed 0), mean held-out NLL, maximum-likelihood fits (BFGS).
Candidates (CV NLL): Normal with polynomial mean (deg 2-6) and polynomial log-sigma (deg 1-4); Student-t variants.
- log-sigma linear: ~0.80-0.82 (bad); quadratic: ~0.615 for mean deg 3-5; deg-2 mean: 0.654.
- Student-t gave no gain (0.615 vs 0.616), so Normal kept.
- cubic mean + quartic log-sigma 0.608, but within noise of quadratic; chose the simpler one (cubic mean, quadratic log-sigma, Normal, 7 constants).
Final: refit on all data (train NLL 0.593). Expect a bit worse than the CV figure.
expected_nll: 0.62
