# Notes
Data: 1 covariate x0 in [0,1]. Mean of y is wiggly in x0; for x0 > ~0.65 y splits into two branches
(a lower one continuing the wiggle, an upper one rising linearly), with heteroscedastic spread.

Program: 2-component Normal mixture. Component A: mean = degree-6 polynomial in x0, sd = exp(quadratic).
Component B: mean linear, sd = exp(linear). Weight of B = sigmoid(a*(x0-c)). ~21 constants.
Fit by ML (BFGS, multi-start) on all data.

Validation: 5-fold CV (refitting everything per fold). Mean held-out NLL:
poly degree A 5/6/7/8 -> 0.190/0.182/0.182/0.182 (B linear or quadratic similar; B constant much worse 0.416).
Chose degree 6 for A, linear B. In-sample NLL 0.160; CV NLL ~0.182.

expected_nll: 0.185
