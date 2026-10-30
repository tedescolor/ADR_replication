# Notes
Data: 1 covariate, n=250. Mean of y is non-monotone in x0; at x0>0.7 y splits into two branches
(upper rising ~1.2->2.5, lower falling ~0->-1), spread grows with x0.

Model: 2-component Normal mixture. Lower component: degree-6 polynomial mean, log-sd linear in x0.
Upper component: quadratic mean, constant sd. Mixture weight of the upper component = sigmoid(linear in x0).
Fitted by maximum likelihood (L-BFGS, 3 starts). 21 constants, well under the limits.

Validation: 5-fold CV, 2 repeats, refitting each fold. Mean held-out NLL by (lower deg, upper deg):
(3,1) .343, (4,1) .334, (5,1) .325, (4,2) .340, (5,2) .318, (6,2) .304, (7,2) .318, (6,3) .318.
Chose (6,2). Train NLL on all data is 0.239. The CV figure is slightly optimistic because
it was also used to choose among 8 configurations, so I add a small margin.

expected_nll: 0.33
