# Notes
Data: 1 covariate x in (0,1). Binned exploration: mean of y is nonlinear in x; for x>0.65 y becomes bimodal
(an upper branch rising to ~2.5 and a lower branch falling), with spread growing in x.

Model: 2-component Gaussian mixture. Component A mean = cubic in x, B mean = quadratic in x;
log-sd of each linear in x; weight of B = sigmoid(linear in x). 13 fitted parameters (plus structural constants), fit by ML (BFGS, multi-start).

Validation: 5-fold CV (random folds), refitting everything per fold. Compared polynomial degrees (A,B):
(4,2) 0.384, (4,3) 0.262, (5,2) 0.369, (5,3) 0.263, (6,3) 0.263 (coefficient counts, i.e. cubic/quadratic = 4,3).
Chose the smallest adequate (cubic, quadratic): CV NLL 0.262; in-sample 0.252.
Contract check passes.

expected_nll: 0.265
