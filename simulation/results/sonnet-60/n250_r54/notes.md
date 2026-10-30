# Notes
Data: 250 rows, one covariate x0 in [0,1]. Mean of y is non-monotone in x0 (dips near 0.25, rises after 0.5); spread grows strongly with x0.
Model: Normal(cubic(x0), exp(quadratic(x0))), fit by max likelihood. 7 constants.
Validation: 3x repeated 10-fold CV of mean NLL. Cubic mean/quadratic log-sd: 0.650 (Normal), same with StudentT (no gain).
Linear log-sd: 0.75; quartic/quintic mean: 0.655/0.657; cubic log-sd: 0.652; quadratic mean: 0.73.
Training NLL 0.626; CV is the honest estimate.
expected_nll: 0.65
