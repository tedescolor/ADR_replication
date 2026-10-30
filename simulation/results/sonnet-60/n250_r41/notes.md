# Notes
Data: 1 covariate x0 in [0,1], y with nonlinear mean in x0 and strongly increasing spread.
Validation: 10-fold CV repeated with 3 seeds (mean NLL), refitting by ML on each training fold.
Tried: Normal / StudentT, polynomial mean degree 2-5, polynomial log-sd degree 1-3, and a skew-normal variant.
Results (CV NLL): Normal cubic mean + quadratic log-sd 0.623; degree 4 mean 0.621 (noise level); StudentT no gain; skew-normal 0.622.
Chosen: Normal(cubic mean, exp(quadratic)) -- 7 fitted constants, simplest model at the CV plateau.
In-sample NLL 0.591; CV 0.623.
expected_nll: 0.63
