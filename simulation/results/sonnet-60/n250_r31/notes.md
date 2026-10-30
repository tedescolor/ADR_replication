# Notes
Data: 250 rows, one covariate x0 in [0,1]. Mean of y is nonlinear in x0 (dip near 0.2-0.3, rise after 0.5); spread is heteroscedastic (small ~0.3 for x0<0.5, large ~1.5 near x0=1).

Tried: Normal and StudentT, polynomial mean (degree 3-6), polynomial log-sd (degree 1-3), fit by ML (BFGS).
Validation: 3x repeated 10-fold CV, mean held-out NLL.
Results: Normal, cubic mean + quadratic log-sd = 0.646 (best); log-sd linear ~0.75; StudentT and higher degrees no better (0.649-0.66).
Final: Normal(cubic mean, exp(quadratic log-sd)), 7 fitted constants (plus a few structural), refit on all data. Train NLL 0.623.

expected_nll: 0.65
