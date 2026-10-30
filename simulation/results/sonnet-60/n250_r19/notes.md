# Notes
Data: 1 covariate x0 in (0,1). Mean of y is nonlinear; for x0>0.7 y splits into two branches (upper rising ~1.5->2.6, lower falling to -1).
Model: 2-component Normal mixture. Component 1 mean cubic in x, component 2 mean linear in x, log-sd linear in x for each, mixing weight logistic-linear in x. 17 constants.
Validation: 5-fold CV (shuffled) of ML fits with random restarts. Compared polynomial degrees (cubic..sextic lower branch, linear/quadratic upper, linear/quadratic weight). Cubic/linear/linear had CV NLL 0.181; higher-order variants 0.178-0.23 (no reliable gain, more overfit risk). Train NLL 0.137.
expected_nll: 0.19
