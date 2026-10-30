# Notes
Data: 1 covariate x0 in [0,1]; y mean is nonlinear in x0 and, for x0>~0.65, the conditional density becomes bimodal (a lower branch continuing the smooth curve, an upper branch near 1.4-2.4).
Model: 2-component Gaussian mixture. Means are polynomials in (x0-0.5) (degree 5 and 2), log-scales linear in x0, the weight is sigmoid of a quadratic in x0. 20 fitted parameters (all constants written in). Fit by ML (BFGS, 4 restarts).
Validation: 5-fold CV over the whole pipeline (refit per fold). Mean held-out NLL: degrees (3,1) 0.263, (4,1) 0.227, (5,1) 0.220, (5,2) 0.208, (6,2) 0.210. Chose (5,2). In-sample NLL 0.196. The degree was chosen on the same CV folds, so the estimate is slightly optimistic.
expected_nll: 0.215
