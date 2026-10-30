# Notes
Data: 1 covariate, y mean is nonlinear (cubic-like dip then rise), sd grows strongly with x0.
Candidates (10-fold CV, repeated, mean NLL): Normal with polynomial mean (deg 2-6) and polynomial log-sd (deg 1-3);
same with Student-t. Best: cubic mean, quadratic log-sd, Normal: CV NLL ~0.637 (deg-2 mean 0.741, deg-4+ ~0.639-0.645,
linear log-sd 0.752). Student-t gave no gain. Final fit on all 250 rows; 7 constants.
Did not explore skewed/mixture components for the high-x tail.
expected_nll: 0.65
