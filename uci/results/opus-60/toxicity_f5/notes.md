# Notes

## Data
728 rows, 6 covariates. x3, x4 are jittered integer counts (mostly 0). y > 0, right-skewed;
linear OLS residual sd ~0.97, residuals heavy-tailed (excess kurtosis ~3) and clearly
heteroscedastic (larger spread at both extremes of x5 and x0, and at large x1).
log(y) is worse (strong left tail), so I modelled y directly.

## Validation
10-fold CV repeated twice (different seeds), mean held-out NLL; all parameters fitted by ML
(L-BFGS, tiny L2 1e-3 on slopes) on the training folds only. Features use centered covariates.

## Results (CV NLL)
| model | NLL |
|---|---|
| Normal, linear mean, const scale | 1.409 |
| Normal, cubic mean, log-scale linear + x0^2 + x5^2 | 1.278 |
| SkewNormal, same | 1.269 |
| 2-normal mixture, same | 1.243 |
| Student-t, same | 1.245 |
| t, cubic mean + 6 interactions, scale + x0*x5 | 1.214 |
| t, scale as quadratic function of fitted mean | 1.226-1.285 (worse) |
| **t, cubic + 9 interactions (incl. x3) mean, scale lin + x0^2,x5^2,x0*x5** | **1.196** |
| t, full quadratic interactions (15) + cubes | 1.197 |
| adding more scale interactions | 1.197-1.204 |

Student-t (df ~4.7) beat Normal/SkewNormal and matched the 2-component mixture with fewer
constants. Heteroscedastic log-scale was the biggest gain.

## Final program
StudentT(df, mu(x), exp(ls(x))): mu = cubic polynomial in each covariate + 9 pairwise
interactions (28 coefs); ls = linear + x0^2 + x5^2 + x0*x5 (10 coefs); 1 df; 4 centering
constants: 43 constants total. Fitted on all data (train NLL 1.131). Checked extrapolation to
x0 = 5.92 (domain max, beyond data max 5.15): mean/scale stay sensible.

The CV estimate is 1.196; adding a little for model-selection optimism:

expected_nll: 1.21
