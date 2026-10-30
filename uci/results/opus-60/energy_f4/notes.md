# Notes

## Data
7654 rows, 4 covariates, y in about [420, 495]. y is driven mostly by x0 (r=-0.95) and x1 (r=-0.87).
A linear model leaves residual sd 4.55. The residuals are left-skewed (skew -0.57) and heavy-tailed
(excess kurtosis 4.4), and their spread varies with x. kNN shows strong local structure that
smooth polynomials miss: a 5-NN in weighted (x0,x1,x2,x3) gives residual sd 3.70, against 4.14 for
a cubic polynomial. Most of this is in the (x1,x2) direction.

## Validation
5-fold CV (fixed random fold assignment, seed 0) on mean held-out NLL in the original y units.
Each model is refit per fold, with 3 random restarts per fold; the best restart is chosen on
training loss.

## Results (CV NLL)
| model | #params | CV NLL |
|---|---|---|
| poly2 mean, Normal, const sd | 17 | 2.864 |
| poly2, Student-t, log-linear sd | 22 | 2.842 |
| poly2, 2-Normal mixture, log-linear sd | 23 | 2.833 |
| poly3, 2-Normal mixture, const sd | 39 | 2.815 |
| MLP H=4 tanh mean, 2-mix, log-lin sd | 33 | 2.801 |
| MLP H=6, 2-mix, const sd | 41 | 2.801 |
| MLP H=6, 2-mix, log-lin sd | 45 | 2.788 |
| MLP H=8, 2-mix, log-lin sd | 57 | 2.784 |
| MLP H=8, 2-mix, log-lin sd, logistic-linear weight | 61 | 2.777 |
| MLP H=10, 2-mix, log-lin sd | 69 | 2.769 |
| MLP H=12, 2-mix, log-lin sd (81 consts, over budget) | 81 | 2.764 |
| **MLP H=11, 2-mix, log-lin sd, logistic-linear weight** (weight decay 1e-5) | 79 | **2.761** |
| same, weight decay 1e-4 | 79 | 2.759 |
| same, weight decay 1e-3 | 79 | 2.772 |

## Final program
- Mean: m = c + sum_k v_k tanh(w_k . x + b_k), with 11 tanh units on the raw x. The
  standardization is folded into the weights, so it costs no constants.
- Noise: a 2-component Normal mixture, sigmoid(a) N(m, s) + sigmoid(-a) N(m + d, s*g). Both
  log s and a are linear in x. The second component is a wide, left-shifted tail component
  (d ≈ -5.4, g ≈ 2.5).
- 79 constants in total. The model passes the contract check. s stays in [1.9, 7.4] over the
  whole covariate box, and tanh keeps the mean bounded, so the model behaves safely when
  extrapolating.
- Fit by maximum likelihood (L-BFGS with analytic gradients, weight decay 1e-5) on all
  data. Restarts vary by up to 0.013 in training NLL (2.744 to 2.761), so I ran 15 restarts and
  kept the best (training NLL 2.7442). Decay 1e-4 had essentially the same CV, but its
  best of 8 full-data restarts reached only 2.7462 in training NLL, so I kept 1e-5. Evaluating the
  emitted model.py reproduces the fitted NLL exactly. Over the covariate box, the 5%/95%
  quantiles stay within [401, 518].

Why this one: it has the best CV NLL among the in-budget programs. The gain from extra
hidden units was still steady at the budget limit, while the noise-model terms
(heteroscedastic scale, x-dependent mixture weight) each bought about 0.01.

The CV estimate is 2.761, from models trained on 80% of the data. The final fit uses all the
data, which should help a little; selecting among models adds a little optimism. Net
estimate:

expected_nll: 2.763
