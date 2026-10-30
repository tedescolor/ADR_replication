# Notes
Data: 1 covariate x0 in [0,1], y has a nonlinear mean in x0 (dips, then rises) and a spread that grows strongly with x0.
Candidates: Normal / StudentT / SkewNormal, with the mean a polynomial (degree 2-4) in x0 and log-sd a polynomial (degree 1-2) in x0. All were fit by maximum likelihood.
Validation: 5-fold CV (one random split), mean held-out NLL. Selection used the same folds, so the best figure is slightly optimistic.
Results (CV NLL): Normal cubic-mean/quadratic-log-sd 0.608; Normal quartic/quadratic 0.605; StudentT cubic/quadratic 0.608; SkewNormal 0.61-0.63; linear log-sd 0.66-0.78.
Final: Normal, cubic mean, quadratic log-sd (7 fitted constants), refit on all data. The quartic mean and the heavier-tailed families did not beat it meaningfully, so I took the simpler one.
Train NLL 0.583; CV NLL 0.608.
expected_nll: 0.62
