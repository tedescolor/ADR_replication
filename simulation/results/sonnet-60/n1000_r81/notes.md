# Notes
Data: 1 covariate x0 in [0,1], y with strongly nonlinear mean (dip near 0.2, rise to ~0.8) and sd that is smallest near 0.4 and grows sharply toward x0=1.
Validation: 5-fold CV (fixed shuffle), parameters fit by ML on each training fold, mean held-out NLL.
Candidates (CV NLL): Normal poly-mean(deg 4-6)/log-sd poly(deg 1) ~0.765; log-sd deg 2 ~0.62; deg 3 ~0.62; deg 4 ~0.609-0.611.
StudentT and SkewNormal heads gave no gain (same or slightly worse), so residuals are close to Gaussian given the heteroscedastic fit.
Final: Normal(mu(x), exp(ls(x))), mu quintic polynomial, ls quartic polynomial (Horner form), 11 coefficients fit on all data; CV NLL 0.6092 (train NLL 0.601).
expected_nll: 0.61
