# Notes

## Data
726 rows, 6 covariates. x3 and x4 are near-integer counts (mostly 0) plus tiny noise; x1 is
clustered around a few values. Linear OLS residuals: skewed (0.36), heavy-tailed (excess kurtosis 3.2),
and heteroscedastic (spread is U-shaped in x5 and larger at extremes of x0/x1).

## Validation
10-fold CV, repeated 2-3 times with different shuffles; every model is refit by maximum likelihood
(L-BFGS) on each training fold, and I report the mean held-out NLL per row.

## Candidates (CV NLL)
- Normal, linear mean, constant scale: 1.373
- Student-t, linear mean, constant scale: 1.318
- t, linear mean, log-scale linear in x + x5^2: 1.259 (skew-t: 1.250)
- log(y) models (with Jacobian): 1.23-1.28, worse than the original scale
- t, mean + hinge splines: 1.237; mean + all pairwise interactions + min(x3,1), min(x4,1): 1.210
- pruned mean (x0,x1,x2,x4,x5, min(x3,1), min(x4,1), x0x1, x0x2, x0x5, x1x2):
  scale lin+x5^2: 1.207; scale + x0^2, x1^2: 1.200; **scale + x0^2, x1^2, x0x5: 1.191 (chosen)**
- skew-t gave no gain over t once the mean had interactions (1.2071 vs 1.2071).

## Final program
Student-t (df = exp(1.711) ≈ 5.5). Mean has 12 terms; log-scale is a quadratic with 11 terms,
clipped to [-1.5, 1.8] so it stays sensible when extrapolating to the corners of the covariate box (the
training range was [-1.23, 1.55]). 28 constants. In-sample NLL 1.153, CV NLL 1.191.

I estimate a mean NLL of about 1.20 on new data: the CV figure plus a small allowance for having
picked the best of about 20 candidates on the same folds.

expected_nll: 1.20
