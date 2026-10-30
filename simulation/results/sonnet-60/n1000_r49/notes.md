# Notes
Data: 1 covariate x0 in [0,1]; mean is a nonlinear (cubic-like) function of x0, spread grows strongly for x0>0.6.
Tried (10-fold CV, NLL): Normal / StudentT / SkewNormal with polynomial mean (deg 3-6) and polynomial log-scale (deg 2-3).
Best/simplest: Normal, cubic mean, quadratic log-sd: CV NLL 0.5902 (t, skew-normal, higher degrees were no better; lower degrees clearly worse).
Final: that model fitted on all data (train NLL 0.584), 7 constants.
expected_nll: 0.592
