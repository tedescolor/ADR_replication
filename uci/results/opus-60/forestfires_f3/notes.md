# Notes

## Data
- y: 413 rows, heavily right-skewed (median 1.0, max 1092). About 23% of values are negative
  (down to -1.37) and there is a bump around 0, so y looks like "zero or a positive heavy-tailed
  amount", plus additive noise.
- Covariates: no strong signal. Spearman correlations with y, with P(y>2) and with log y | y>2
  are all |rho| < 0.16. Grouping the near-integer covariates x0..x3 shows nothing consistent.

## Validation
10-fold CV, repeated 2–5 times with different seeds. Every fold refits by maximum likelihood
(L-BFGS-B, several starts). The score is the mean held-out NLL on the original y scale.

## Marginal mixtures (CV NLL)
| model | train | CV |
|---|---|---|
| Normal + Gamma | 3.020 | 3.054 |
| Normal + Student-t | 3.042 | 3.060 |
| Laplace + LogNormal | 2.921 | 2.930 |
| Normal + LogNormal | 2.909 | 2.922 |
| Student-t + LogNormal | 2.909 | 2.922 (df goes to infinity) |
| Normal + Normal + LogNormal | 2.906 | 2.923 |
| Normal + 2 LogNormal | 2.906 | 2.932 |
| **SkewNormal + LogNormal** | 2.907 | **2.920** |

## Covariate effects
Starting from Normal+LogNormal, I tried one standardized covariate at a time, linear in the
mixture logit, in the lognormal mu, or in the lognormal log-sigma: 36 candidates in all. None
gave a clear improvement:
- Best: x8 in log-sigma, -0.0095 ± 0.0085 (paired SE, 5x10-fold).
- x0 in the weight: -0.0015 ± 0.005.
- x11 (outlying values) hurt badly, +0.17 to +0.21.

With 413 rows, those small gains are noise, so the final model is unconditional.

## Final program
Mixture of two components (6 constants):
- 0.392 × SkewNormal(loc -0.715, scale 0.880, shape 2.07)
- 0.608 × LogNormal(mu 1.455, sigma 1.821)

Train NLL is 2.907 and CV NLL is 2.920. Because y is so heavy-tailed, the score on a new
test set will vary a lot (one huge value can move the mean). I add a small margin to the CV
estimate.

expected_nll: 2.93
