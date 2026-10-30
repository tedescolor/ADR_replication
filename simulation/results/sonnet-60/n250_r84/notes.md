# Notes
Data: 250 rows, 1 covariate. Mean is nonlinear in x0 (cubic-like), spread is U-shaped (small near x0~0.3, large near x0=1).
Candidates (10-fold CV mean NLL): Normal / StudentT, polynomial mean degree 3-6, polynomial log-sd degree 1-3.
Best: Normal, cubic mean, quadratic log-sd: CV NLL 0.566 (StudentT and higher degrees were worse, 0.57-0.59; constant-ish log-sd 0.74).
Final: Normal(cubic(x-0.5), exp(quadratic(x-0.5))), fit by ML on all data. 7 constants.
Estimate is slightly optimistic as the model was selected on the same CV.
expected_nll: 0.575
