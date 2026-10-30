# Notes
Data: 1 covariate x0 in [0,1]; y mean is a smooth nonlinear (cubic-like) function of x0, sd is U-shaped (min near 0.5, large near 1); residual skew small.
Validation: 5-fold CV (seed 0) of Normal/StudentT with polynomial mean (deg 3-6) and polynomial log-sd (deg 1-3).
Results (CV NLL): log-sd linear ~0.745; quadratic ~0.615-0.617; cubic ~0.617-0.619. Mean degree 3 vs 6 differ <0.002; Student-t gave no gain.
Final: Normal(cubic(x0), exp(quadratic(x0))), 7 constants-fit parameters (4 mean, 3 log-sd), fit by ML on all data.
Train NLL 0.611; CV NLL ~0.617.
expected_nll: 0.62
