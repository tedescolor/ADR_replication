# Notes
Data: 1 covariate x0 in [0,1]. Scatter plot shows two branches above x~0.5 (upper roughly linear, lower curved), spread growing with x.
Model: 2-component Gaussian mixture. Lower branch mean = quartic in (x-0.5), upper branch mean = quadratic;
log-sd linear in x per component; upper-branch weight = sigmoid(linear in x). 14 fitted parameters, ~25 literals.
Fitted by ML (BFGS, multi-start) on all 1000 rows.
Validation: 5-fold CV of the whole fitting procedure (mean NLL):
(mean deg 3/1) 0.278, (3/2) 0.281, (4/1) 0.268, (4/2) 0.261 <- chosen; constant sd 0.44 / 0.34 (much worse); lower-branch sd constant 0.286.
Train NLL of final fit 0.250. Not tried (time): other families (StudentT), 3-component mixtures.
expected_nll: 0.27
