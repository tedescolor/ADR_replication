# Notes
Data: 1 covariate x0 in [0,1]. y has one branch for x<0.45 (U-shaped mean), then splits into two
branches (rising ~ to 2.5, falling ~ to -1) with heteroscedastic noise.
Model: 2-component Gaussian mixture; component means are cubic / quadratic in x, log-sds linear in x,
mixing weight = sigmoid(quadratic in x). 14 fitted constants plus a few structural ones.
Fit by maximum likelihood (L-BFGS, multiple restarts).
Validation: 5-fold CV (shuffled) comparing polynomial degrees (mean, mean, logsd, logsd, logit w):
(3,1,1,1,1) 0.285; (3,2,1,1,2) 0.180; (4,1,2,1,2) 0.261; (4,2,2,2,3) 0.183; (5,2,2,1,3) 0.184; (3,1,0,0,2) 0.274.
Chose (3,2,1,1,2), the simplest of the best group. Full-data training NLL 0.168.
expected_nll: 0.18
