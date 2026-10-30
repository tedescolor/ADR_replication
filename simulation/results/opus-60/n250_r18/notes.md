# Notes

Data: 250 rows, x0 in [0,1], y shows two branches (an increasing ~linear branch for x>0.4
reaching 2.5 at x=1, and a wavy lower branch), with heteroscedastic noise.

Models: mixtures of K Gaussians, means polynomial in t=2x-1, log-sd polynomial, mixture
logits polynomial; fitted by MLE (L-BFGS, analytic gradients, small L2, random restarts).

Validation: 5-fold CV repeated twice (refit incl. restarts inside each fold), mean held-out NLL.
- K=1 (deg5 mean): train 0.70
- K=2, mean deg5, log-sd deg1, logit deg1: CV 0.252  (train 0.131)
- other K=2 variants (mean deg/log-sd deg/logit deg): 4/1/1 0.302, 6/1/1 0.272, 7/1/1 0.275,
  5/2/1 0.335, 5/1/2 0.267, 5/0/1 0.405
- K=3 (mean/log-sd/logit deg): 3/1/1 0.365, 5/1/1 0.341, 4/1/2 0.363, 6/1/1 0.386,
  5/2/2 0.431, 7/2/2 0.441 (train lower, but unstable/overfits)
- K=4, 5/1/1: CV 0.469

Final: K=2, deg5 means, linear log-sd and logit (21 numeric constants by the contract count). The CV estimate is used
as the expected NLL (slightly inflated for safety as the full-data fit uses 25% more data
but random-restart optimisation can land in different optima).

expected_nll: 0.26
