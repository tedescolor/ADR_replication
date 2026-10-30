# Notes

## Data
- 413 rows, 12 continuous covariates with jitter added, and a target y that is very heavy-tailed and right-skewed (median 1.0, max 1092).
- About 100 values of y are negative, all between -1.4 and 0. They look like a point mass at 0 with
  N(0, ~0.6) jitter added, which probably accounts for ~45% of the rows. The rest of the y values are positive and roughly lognormal.

## Model families tried (all as mixtures; NLL is on the original y scale)
Validation: 5-fold CV repeated 3 to 5 times with different shuffles. All parameters were refit on each training fold.
| model | params | in-sample | CV NLL |
|---|---|---|---|
| N(0,s) + LogNormal | 4 | 2.960 | **2.969** |
| StudentT(0,s) + LogNormal | 5 | 2.960 | 2.969 (df goes to infinity, so it is the same as the Normal model) |
| N(0,s) + 2 LogNormals | 7 | 2.958 | 2.978 |
| N(0,s) + LogNormal + extra Normal | 7 | 2.953 | 2.984 |
| N(0,s) + Gamma | 4 | 3.075 | 3.097 |

## Covariates
I added each covariate on its own (standardized; log transforms for the skewed x7 and x11) as a
linear effect in one of three places: the logit of the zero weight, the lognormal mu, or the log of the lognormal sigma.
I compared against the base model's CV of 2.970 under the same splits.
No change was larger than 0.004 nats, which is well inside noise (the SE of the mean NLL is about 0.10).
Most changes made the score worse. I conclude the covariates carry no usable signal, so the model ignores x.

## Final program
Mixture of 0.4495 * Normal(0, 0.6308) and 0.5505 * LogNormal(mu=1.7891, sigma=1.6446). It has 6 constants.
Its in-sample NLL is 2.960 and its CV NLL is 2.969. The heavy tail makes new-data NLL noisy (per-row SE about 0.1).
I add a small allowance for this.

expected_nll: 2.98
