# Conditional density model — notes

## Data
7654 rows, 4 covariates, y in [420, 496]. y is strongly (negatively) related to x0, then x1; x2 and x3
contribute less but matter. Linear-OLS residual SD 4.6, residuals left-skewed (skew -0.43) and
heavy-tailed (excess kurtosis 3.7). kNN (k=5) leave-one-out residual SD ~3.5 shows fine local structure
that a compact formula cannot fully capture.

## Validation
5-fold CV (fixed random fold assignment), mean held-out NLL on the original y scale; all parameters
refit on each training fold (multi-start L-BFGS, best training objective kept).

## Candidates (5-fold CV NLL)
| model | #params | CV NLL |
|---|---|---|
| quadratic mean, log-linear scale, Normal | 20 | 2.874 |
| same, Student-t | 21 | 2.852 |
| same, SkewNormal | 21 | 2.863 |
| same, 2-normal mixture | 27 | 2.838 |
| cubic mean, Normal | 40 | 2.848 |
| cubic mean, Student-t | 41 | 2.820 |
| cubic mean, 2-normal mixture | 47 | 2.809 |
| tanh-MLP H=6 mean, Normal | 42 | 2.831 |
| MLP H=6, 2-mixture | 49 | 2.790 |
| MLP H=8, 2-mixture | 61 | 2.788 |
| MLP H=10, 2-mixture | 73 | 2.772 |
| **MLP H=8, 2-mixture, log-scale also uses hidden units** | 69 | **2.770** |
| MLP H=9, same | 76 | 2.768 |

(Fold-to-fold SD of the NLL is ~0.02; differences < 0.005 are noise, and part of the spread
comes from local optima of the net.)

## Final program
- 8 tanh hidden units of the raw covariates (standardisation folded into the weights).
- mean m = linear combination of the hidden units.
- scale s = exp(linear in x + linear in hidden units).
- Mixture of N(m, s) and N(m + d*s, r*s) (d ≈ -2.3, r ≈ 3.2: a wider, lower-shifted component
  that produces the left skew / heavy tail), with mixing weight sigmoid(linear in x).
- 69 constants; passes the contract check. Fit on all data, best of 6 random starts; training NLL 2.754.
H=8 with hidden units in the scale was chosen over H=9: same CV score within noise, fewer constants.
I checked the box corners: the density stays valid and plausible there (tanh units are bounded).

## Expected performance
CV NLL 2.770; the full-data fit has slightly more data, but there is also some optimiser/selection noise, so:

expected_nll: 2.775
