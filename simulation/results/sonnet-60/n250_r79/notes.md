# Notes
Data: 250 rows, 1 covariate. Mean of y is nonlinear in x0 (dips near 0.25, then rises); spread is
heteroscedastic (small around 0.3-0.6, large at x0>0.8).
Validation: 10-fold CV (single random split, seed 0), mean NLL. Candidates: Normal / StudentT,
polynomial mean degree 2-5, polynomial log-sd degree 1-3.
Results (CV NLL): deg2/1 0.882; deg3/1 0.761; deg3/2 0.609 (best); deg4/2 0.611; deg3/3 0.610;
deg5/2 0.613. StudentT gave no gain (df large), so Normal was kept.
Final: Normal, cubic mean, quadratic log-sd, fit by ML on all data (7 params). Train NLL 0.577.
Chosen as simplest model at the CV optimum.

expected_nll: 0.61
