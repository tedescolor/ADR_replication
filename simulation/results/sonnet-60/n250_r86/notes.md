# Notes
Data: 250 rows, 1 covariate in [0,1]. y has a non-monotone mean (dip near x=0.25, rises after 0.5) and spread that grows strongly with x.
Validation: 3x repeated 5-fold CV, with MLE refit on each training fold.
Candidates (CV mean NLL):
- Normal, polynomial mean (degree 3-6) with log-sd linear: 0.79-0.81
- Normal, log-sd quadratic: degree 3 mean 0.693, degree 4 0.698, degree 5 0.701, degree 2 0.730
- Normal, cubic mean, log-sd cubic/quartic: 0.692 / 0.684 (within noise, so not adopted)
- StudentT: df went to infinity, identical to Normal
- SkewNormal: 0.726, worse
Final: Normal(cubic(x), exp(quadratic(x))), fit by MLE on all data. 7 constants.
Train NLL is 0.664 and CV NLL is 0.693. I report the CV figure.
expected_nll: 0.70
