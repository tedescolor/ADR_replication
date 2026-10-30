# Notes
Data: 250 rows, 1 covariate. Mean is nonlinear in x0 (dip near 0.2, rise to ~0.9), sd grows strongly with x0 (0.2 -> 1.5).
Tried (10-fold CV, mean NLL): Normal / StudentT / SkewNormal with polynomial mean (deg 2-6) and polynomial log-sd (deg 1-3).
Best: Normal, cubic mean, quadratic log-sd: CV NLL 0.656. Student-t and skew-normal gave no gain; higher degrees slightly worse; quadratic mean 0.754.
Final: that model refit on all data (8 constants in the polynomials; 7 free params).
Train NLL 0.623; CV NLL 0.656.

expected_nll: 0.66
