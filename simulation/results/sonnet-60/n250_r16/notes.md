# Notes
Data: 250 rows, one covariate x0 in [0,1]. Mean is strongly nonlinear in x0; spread varies with x0 (sd ~0.3 for x0<0.6, ~1.4 near x0=1).
Candidates: Normal and StudentT, polynomial mean (deg 2-6) x polynomial log-sd (deg 0-4), fit by ML (BFGS).
Validation: single 10-fold CV (random folds, seed 0), mean held-out NLL. Best: mean deg 4, log-sd deg 4, CV NLL 0.577
(deg 5/4: 0.580; 4/3: 0.593; 2/1: 0.815). StudentT df went to infinity, so Normal is used. Standardized residuals: skew -0.05, excess kurtosis -0.44.
Final: Normal(poly4(x), exp(poly4(x))), 10 constants, refit on all data. Degree choice was selected on the same CV, so slight optimism is expected.
expected_nll: 0.60
