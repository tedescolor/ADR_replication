# Notes

## Data
- `y` is a mixture: tight spikes (all with sd about 0.020) at 0, 0.5, 1, 1.163, 2, 3, 4, 5, 6, 7, which hold about 80% of rows (zero alone is 34%), plus a continuous part on about [1.6, 5].
- Covariates: x0 (age-like, and x1 is almost the same column), x5 (skewed, 2..343), x6, x19 (a count),
  x20 (skewed), and 20 noisy binary indicators. No covariate strongly predicts y on its own. The main signal is which spike or
  continuous component a row falls in, and that depends on log(x5) differently for each component
  (small x5 favours the 0.5 spike and the continuous part; large x5 favours 0 and 1).

## Model family
A mixture of 13 normals: 10 spikes with a shared sd and fitted locations, plus 3 continuous normals.
The mixture logits are linear in the covariates. They are fitted by maximum likelihood (L-BFGS with an analytic gradient
and a small L2 penalty on the slopes; final value 1e-4, chosen by CV over 1e-4 / 1e-3 / 3e-3 = 0.1864 / 0.1870 / 0.1923 on a second split), and then written into model.py as nested `Mixture`s
(3 sub-mixtures plus a top level whose weights are the sums of the unnormalised sub-weights).

## Validation
5-fold CV (one fixed permutation) with the mean held-out NLL. All comparisons use the same folds.
| model | #slope coefs | CV NLL |
|---|---|---|
| marginal (no covariates), 3 continuous comps | 0 | 0.287 |
| 2 logit groups (zero vs rest), all 25 feats | 25 | 0.261 |
| 3 groups | 50 | 0.248 |
| 4 groups (0 / low spikes / 2,3 / 4-7 / cont), all feats | 100 | 0.227 |
| 4 groups, 11 selected feats | 44 | 0.228 |
| 5 groups (2.3-ish continuous comp separate), 7 feats | 35 | 0.205 |
| rank-1 per-component loadings, 11 feats | 23 | 0.200 |
| hybrid: per-component log(x5), 5-group for age,x19,x2,x3,x4,x22 | 42 | 0.192 |
| **hybrid: per-component age & log(x5), 5-group for x19,x2,x3,x4,x22** | 49 | **0.187** |

The final model is the last row. It uses exactly 80 constants and 485 AST nodes. On a second CV split it scored 0.187 (fold SE 0.006) with L2 1e-3, and 0.1864 with the final L2 of 1e-4.
I also tried per-component x19 instead of x4 plus finer groups: CV 0.190, which is worse. The budgets on literals and AST nodes
were the binding constraints, which is why more per-component effects were not added.
Rounding the constants to 4 significant digits changes the NLL by about 0.0002 (checked by evaluating model.py itself).

## Estimate
The CV NLL is about 0.187 (fold-to-fold SE about 0.02; the differences between models are paired and much tighter).
The final fit uses all the data (slightly better than the 80% fits in CV), but the model was selected on the same CV,
so a small optimism allowance applies.

expected_nll: 0.19
