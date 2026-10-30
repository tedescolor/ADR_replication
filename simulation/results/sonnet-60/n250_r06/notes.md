# Notes
Data: 1 covariate x0 in [0,1], y with nonlinear (cubic-like, wavy) mean and variance that is U-shaped in x0 (large at x0 near 1).
Validation: 10-fold CV (single shuffle), all parameters refit on each training fold, mean held-out NLL.
Candidates (CV NLL): cubic mean + linear log-sd Normal 0.758; quartic 0.762; Student-t no gain; 
cubic mean + quadratic log-sd Normal 0.649 (best); quartic/quintic mean 0.653/0.656; t-variant 0.649 (no gain over Normal).
A sine mean was also fit in-sample (0.7365 with linear log-sd) and was no better than the cubic, so not pursued with CV.
Final: Normal(cubic(x0), exp(quadratic(x0))), 7 constants, fitted by ML on all 250 rows.
The CV estimate is slightly optimistic because the model family was selected on the same folds.
expected_nll: 0.66
