# Notes
Data: 1 covariate x0 in [0,1]. Scatter shows a single noisy curve for x<~0.45 (dip then rise), then a split into
two branches (upper rising linearly to ~2.5, lower turning down), with noise sd varying in x.
Model: 2-component Gaussian mixture. Lower mean = quartic polynomial in x; upper mean = lower + hinge
(max(x-c,0)) with linear+quadratic term; log-sd linear in x per component; mixture weight = sigmoid(linear in x).
14 fitted constants (plus a few structural literals), fit by ML (BFGS, Nelder-Mead start).
Validation: 5-fold CV (refitting everything per fold): mean NLL 0.184 (folds 0.12-0.23); in-sample 0.171.
Contract check passes.
expected_nll: 0.19
