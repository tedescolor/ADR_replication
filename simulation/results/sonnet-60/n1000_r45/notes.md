# Notes
Data: 1 covariate x0 in [0,1]; scatter shows a main curved branch plus a second, steeply rising branch appearing for x0>~0.5 (bimodal y at high x).
Model: 2-component Gaussian mixture; component A mean = quartic in (x-0.5), component B mean = quadratic, log-sd linear in x for each, weight of B = sigmoid(linear in x). 14 constants.
Fitted by ML (BFGS). Validation: 5-fold CV (seed 1), refit each fold. CV NLL: quartic/linear 0.235, 5/1 0.232, 6/1 0.226, 4/2 0.219, 5/2 0.220. Chose 4/2 (simple, near best); train NLL 0.211.
Not tried: heavier-tailed components, more mixture components.
expected_nll: 0.222
