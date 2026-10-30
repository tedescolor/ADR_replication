# Notes
Data: 1 covariate x0 in [0,1], y with nonlinear mean (dip near 0.25, rises) and strongly heteroscedastic (sd grows from ~0.3 to ~1.5 at high x).
Candidates (10-fold CV mean NLL, polynomial mean degree 3-6, polynomial log-sd degree 1-3; Normal, StudentT, SkewNormal):
- log-sd linear: ~0.765; log-sd quadratic: ~0.638; higher degrees/other families: 0.638-0.640 (no real gain).
Chosen: Normal, cubic mean, quadratic log-sd (7 constants + none else). Simplest model at the CV plateau.
CV NLL 0.638 (the degree choice was made on the same folds, so slightly optimistic).
expected_nll: 0.645
