# Notes
Data: 1 covariate, y has nonlinear mean in x, and for x>0.7 splits into two branches (bimodal).
Model: 2-component Gaussian mixture; each component has cubic mean in x and log-sd linear in x;
constant mixing weight. 13 fitted constants (plus a few literals). Fit by ML (BFGS, multi-start).
Validation: 5-fold CV of the whole fitting procedure. Compared: single-branch-free variants
(quadratic means 0.51/0.42, cubic mean w/ const sd 0.27, cubic + linear log-sd 0.19-0.22,
quartic means 0.24, logistic-x weights 0.20-0.22 — noisy from local optima). Chosen: cubic, linear log-sd,
constant weight: CV NLL 0.193 (train NLL 0.128). Estimate slightly padded for optimism.
expected_nll: 0.21
