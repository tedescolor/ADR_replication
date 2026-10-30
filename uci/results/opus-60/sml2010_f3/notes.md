# Notes

## Data
2211 rows, 20 covariates. The rows look like a shuffled sensor time series: smooth trajectories in scatter plots, and lag-1 autocorrelation of y in file order is about 0.04. x9 is nearly binary. x10 is bimodal. x3 and x4 have a few large outliers (up to about 590, median about 207). x16, x17 and x18 look like noise.

## Validation
5-fold random CV (seed 0) on mean held-out NLL. Random rather than blocked folds, because the held-out rows are described as coming from the same distribution and the file is shuffled.

## What I tried (CV NLL)
- Constant Gaussian: 3.985
- Linear Gaussian on all 20 covariates: 2.998
- Ridge on all quadratic terms (230 features): 2.713. Too many constants.
- Greedy forward selection of linear and quadratic terms: 2.72 at 30 terms. Gains were slow and spread across many terms.
- Reference only (not submittable): a histogram gradient-boosting model reached RMSE 2.23, about 2.22 Gaussian NLL. Importance: x6 62%, then x5, x19, x2, x0, x3/x4.
- Small tanh MLP, mean with heteroscedastic log-sigma, fitted with L-BFGS and an L2 penalty:
  - 8 inputs, H=8: 2.53
  - 5 inputs (x6, x5, x19, x0, min(x3, 260)), H=10, log-sigma linear in x6 and x19: 2.506 to 2.527 across runs (optimiser noise is about 0.02)
  - Dropping x0 or x3, or swapping in x2 or x1: worse (2.57 to 2.72)
  - Constant sigma: worse by about 0.04
  - Weight decay: 0.01 to 0.1 is best; 5 is too strong
- Student-t noise on the CV residuals: df around 10 gives about 0.005 better NLL than Gaussian.

## Final program
- Mean: 10 tanh units over z = (x6, x5, x19, x0, min(x3, 260)).
- Scale: exp(a + b*x6 + c*x19).
- Noise: StudentT with df = 10. The log-scale offset was re-fitted for df = 10.
- Fitted on all data, best of 4 restarts.
- 76 constants, 390 AST nodes. Passes the contract check.
- In-sample NLL is 2.448. CV NLL for this configuration is about 2.51 to 2.52.

expected_nll: 2.53
