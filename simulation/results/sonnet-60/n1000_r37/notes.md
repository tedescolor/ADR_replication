# Notes
Data: 1 covariate x in [0,1]. Scatter plot shows one curve (dip then rise) for x<0.6 and a split into two branches above: a rising linear upper branch and a falling continuation of the sinusoidal lower branch.
Model: 2-component Gaussian mixture. Lower mean = a + b*sin(w x + p); upper mean linear in x; log-sd linear in x for each; upper weight = sigmoid(linear in x). 13 constants.
Fit by ML (L-BFGS, 15 random restarts). Quadratic upper mean was tried: no gain (CV 0.244 vs 0.238).
Validation: 5-fold CV with refit incl. restarts: NLL 0.238 (in-sample 0.222).
expected_nll: 0.24
