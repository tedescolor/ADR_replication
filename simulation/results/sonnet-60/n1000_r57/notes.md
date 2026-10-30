# Notes
Data: 1 covariate x0 in (0,1). Binned quantiles show y is unimodal at low x (mean dips to ~-0.75 near x=.25) and bimodal at high x: a lower branch continues the smooth curve, an upper branch (rising ~ linearly to 2.5) appears from x~0.6 with growing weight.
Model: 2-component Normal mixture. Component A: quartic mean in x, log-sd linear in x. Component B: linear mean, log-sd linear. Weight of B = sigmoid(linear in x). 13 fitted constants, fit by ML (BFGS, multi-start).
Validation: 5-fold CV mean NLL. Cubic-A/linear-B 0.336; quartic/linear 0.248; quartic/quadratic 0.256; quintic/linear 0.247; quintic/quad 0.250; sextic/quad 0.249. Chose quartic/linear (simplest at the plateau). In-sample NLL 0.232 (0.23 after rounding constants).
expected_nll: 0.25
