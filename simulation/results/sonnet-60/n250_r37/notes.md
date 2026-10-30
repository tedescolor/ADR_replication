# Notes
Data: 1 covariate, y has nonlinear mean (dips then rises) and sd growing strongly with x0.
Validation: 2x repeated 10-fold CV, MLE fits (Nelder-Mead+BFGS), families Normal / StudentT / SkewNormal,
mean polynomial degree 2-5, log-sd polynomial degree 1-3.
Best CV NLL: Normal, cubic mean, quadratic log-sd = 0.638 (t: 0.637, skewnormal: 0.6385 at deg 4/2; no real gain).
Chose simplest: Normal(cubic(x), exp(quadratic(x))), 7 constants. Train NLL 0.613; CV 0.638.
Selection among ~36 configs adds slight optimism, so I add a little margin.
expected_nll: 0.65
