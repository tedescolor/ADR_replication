# Notes
Data: 250 rows, 1 covariate. y mean is nonlinear in x0 (dip near 0.25, rises to ~1 at x0~0.9); spread is U-shaped/increasing at high x0.
Candidates (10-fold CV x3 repeats, mean NLL, parameters refit in each fold): Normal/StudentT/SkewNormal with polynomial mean (deg 2-5) and polynomial log-sd (deg 1-3).
Best: Normal, cubic mean, quadratic log-sd: CV 0.6216 (deg-2 mean 0.711; deg 4/5 worse; log-sd linear 0.816; StudentT df->inf, no gain; skew-normal no gain).
Final: that model fit by MLE on all data (7 constants + structure). Train NLL 0.601.
expected_nll: 0.63
