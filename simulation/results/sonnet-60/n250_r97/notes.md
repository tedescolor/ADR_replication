# Notes
Data: 1 covariate x0 in [0,1]. Mean of y is nonlinear in x0 and for x0>~0.7 the conditional is bimodal
(lower branch continuing the smooth curve, upper branch rising to ~2.5), with growing spread.
Model: 2-component Gaussian mixture. Lower: cubic-polynomial mean, log-sd linear in x. Upper: quadratic mean,
log-sd linear. Weight of upper = sigmoid(linear in x). 13 fitted constants, fit by ML (BFGS, 12 random restarts).
Validation: 5-fold CV (refit incl. restarts each fold): mean NLL 0.173 (folds 0.06-0.30). In-sample NLL 0.097.
Did not compare other families extensively; given the time spent, this structure was accepted. In-sample
is optimistic, so CV is used for the estimate.

expected_nll: 0.18
