# Notes
Data: 1 covariate, y. Single Normal/Student-t with polynomial mean and log-sd: 5-fold CV NLL ~0.637.
Residuals showed bimodality for x>0.6 (a lower branch and an upper branch that splits off).
Final: 2-component Normal mixture; lower comp: quintic mean, quadratic log-sd; upper comp: quadratic mean,
linear log-sd; upper weight = sigmoid(quadratic in x). 5-fold CV (same folds for all candidates, refit per fold):
(3,2,1,1,1) 0.285; (4,2,2,1,1) 0.277; (5,2,2,1,2) 0.254 (chosen); (5,3,2,1,2) 0.253; weight cubic 0.254. Plateau.
Student-t components were not needed. About 25 constants. Fit on full data by ML (L-BFGS, 3 starts).
expected_nll: 0.26
