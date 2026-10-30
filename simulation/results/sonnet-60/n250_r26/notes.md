# Notes
Data: 250 rows, 1 covariate. Mean is nonlinear in x0 (dip then rise); spread grows strongly with x0.
Candidates compared with 3x repeated 5-fold CV (parameters refit per fold, MLE):
polynomial mean (deg 2-5) x polynomial log-sd (deg 1-3), Normal vs StudentT.
Best: cubic mean, quadratic log-sd, Normal (CV NLL 0.709). StudentT gave no gain; deg 4/5 mean or cubic log-sd no gain.
Final: Normal(cubic(x0), exp(quadratic(x0))), fit on all data; 7 constants.
CV NLL is slightly optimistic because the model family was selected on the same folds.
expected_nll: 0.72
