# Notes
Data: 1 covariate, mean of y is non-monotone in x (dips near x~0.25, rises after), sd grows strongly for x>0.6.
Tried Normal/StudentT with polynomial mean (deg 2-5) and polynomial log-sd (deg 1-3), 3x repeated 5-fold CV.
Best: Normal, cubic mean, quadratic log-sd (CV NLL 0.643). Student-t and higher degrees were no better.
Final: fit by ML on all 250 rows; 9 constants. In-sample NLL 0.625.
expected_nll: 0.65
