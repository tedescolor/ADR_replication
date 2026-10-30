# Notes
Data: 250 rows, 1 covariate. Mean of y is nonlinear in x0 (dips near 0.25, rises to ~0.6 by 0.75); spread is U-shaped (small mid, large at high x0).
Candidates, 10-fold CV NLL (same folds): Normal with polynomial mean (deg 3-6) x poly log-sd (deg 1-3); StudentT variants; SkewNormal (constant shape).
- log-sd linear: ~0.80 (bad); quadratic log-sd: 0.631 (cubic mean); deg 4-6 mean slightly worse; cubic log-sd no better.
- StudentT: no gain (0.6308 vs 0.6309). SkewNormal: 0.68-0.70 (overfits).
Final: Normal(cubic mean, exp(quadratic)) fitted by ML on all data; 7 constants.
Train NLL 0.603, CV NLL 0.631.
expected_nll: 0.64
