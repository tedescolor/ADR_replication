# Notes
Data: 250 rows, 1 covariate. Mean of y is nonlinear in x0 (dip near 0.25, rises after 0.4); sd grows strongly for x0>0.7.
Candidates: Normal or Student-t, polynomial mean (deg 3,5,7) x polynomial log-sd (deg 1,2,3), fit by ML.
Validation: 10-fold CV (1 repeat, random folds), mean held-out NLL.
Results: log-sd linear ~0.79; log-sd quadratic ~0.66-0.67 across all mean degrees; Student-t no gain; higher degrees no gain.
Chosen: Normal, cubic mean, quadratic log-sd (7 constants + 0 extras), CV NLL 0.661 (in-sample 0.641).
Caveat: single CV repeat, so differences of ~0.01 between the top candidates are within noise.
expected_nll: 0.67
