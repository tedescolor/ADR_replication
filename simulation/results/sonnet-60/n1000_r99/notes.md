# Notes
Data: 1 covariate x0 in [0,1]. Mean of y is nonlinear in x; for x>~0.6 the conditional
distribution is bimodal (a low branch and a high branch rising to ~2.4), noise grows with x.

Model: 2-component Gaussian mixture. Each component has a polynomial mean in t=x-0.5
(degree 6 and 3) and log-sd linear in t; mixing weight = sigmoid(linear in x). 18 fitted
parameters, fitted by maximum likelihood (L-BFGS). Components partly share the low-x region,
so they are not strictly "branch A / branch B" labelled.

Validation: 5-fold CV (random split, seed 0) of mean NLL, refitting each fold.
Constant sd: 0.28; with linear log-sd: 0.217; degrees (6,3,1,1,1): 0.187;
(7,3,1,1,1): 0.185; (6,3,1,1,2) / (6,3,2,1,1): 0.187. Gains beyond the chosen
config are within noise, so I kept the simpler (6,3,1,1,1). Final fit on all data
(train NLL 0.168). Contract check passes (18 params, well under 80 constants).
Caveat: the config was chosen from ~15 candidates by the same CV, so the estimate is slightly optimistic.

expected_nll: 0.20
