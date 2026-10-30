# Notes
Data: 1 covariate x0 in [0,1]. Exploration showed the mean is nonlinear, noise heteroscedastic, and for x0>0.7 y is clearly bimodal (two branches, ~-0.5 and ~+2).
Model: 2-component Normal mixture; each component has cubic mean in x, log-sd linear in x; mixing weight = sigmoid(quartic in x). 21 constants incl. the "1 -" weight; fitted by ML (L-BFGS, multiple restarts) in work/fit.py.
Validation: 5-fold CV (fixed shuffle), refitting each fold. Mean NLL by (mean deg, logsd deg, weight deg):
(3,1,3) .201; (4,1,3) .210; (4,2,3) .206; (5,2,4) .202; (6,2,5) .205; (2,1,3) .317; (3,0,3) .277; (3,1,2) .200; **(3,1,4) .189**; (3,1,5) .199; (3,1,6) .197; (4,1,5) .197; (3,2,3) .208.
Chose (3,1,4), refit on all data (train NLL 0.172). Selection among ~13 candidates is slightly optimistic, so I add a small margin.
expected_nll: 0.2
