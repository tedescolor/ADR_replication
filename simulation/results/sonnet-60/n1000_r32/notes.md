# Notes
Data: 1 covariate x0 in [0,1]. Mean of y is nonlinear in x0; for x0>~0.65 y becomes bimodal (a lower branch continuing the trend plus an upper branch ~1-2.5).
Model: 2-component Normal mixture; lower mean quartic in x0, upper mean quadratic, log-sd linear in x0 for each, mixing weight sigmoid(linear). 14 constants.
Validation: 5-fold CV (random folds) over degree choices (lower,upper): (3,1) 0.258, (4,1) 0.237, (4,2) 0.231, (5,2) 0.294, (6,2) 0.545 (higher degrees unstable). Chose (4,2). Full-data train NLL 0.220.
expected_nll: 0.235
