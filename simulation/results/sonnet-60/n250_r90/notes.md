# Notes
Data: 250 rows, one covariate x0 in [0,1]. Mean of y is nonlinear (cubic-like), spread is heteroscedastic (small near x~0.25, large near x=1).
Candidates: Normal / StudentT with polynomial mean (deg 2-6) and log-sigma polynomial (deg 1-3). Validation: 3x repeated 5-fold CV, mean held-out NLL.
Results: Normal, mean deg 3, log-sigma deg 2 best at CV NLL 0.585; StudentT no gain; higher degrees worse; mean deg 2 0.694.
Final: Normal(cubic(x0), exp(quadratic(x0))), 7 fitted constants, fit on all data by ML.
CV NLL is slightly optimistic (model selected over ~24 configs), so I add a small margin.

expected_nll: 0.60
