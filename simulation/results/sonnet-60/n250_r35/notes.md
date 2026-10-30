# Notes
Data: 250 rows, 1 covariate. Mean of y is nonlinear in x0 (dip then rise), spread grows strongly with x0.
Candidates (10-fold CV x3 repeats, NLL): Normal with polynomial mean (deg 2-5) and log-sd polynomial (deg 1-4); Student-t variants.
Best: Normal, cubic mean, quadratic log-sd: CV NLL ~0.69 (linear log-sd ~0.82; deg-2 mean ~0.78; t no better; higher degrees no gain).
Final: that model fit by max likelihood on all data, 7 constants (+ few literals), Normal(mu(x), exp(quadratic(x))).
Expected NLL slightly above CV due to selection among candidates.

expected_nll: 0.71
