# Conditional density model for y | x0..x7

## Exploration
- 824 rows, all covariates continuous (look jittered). y in [~2, 83], sd 17.
- Strongest marginal signal: x0 (+), x4 (+), x7 (+, strongly concave -> use log(x7)), x3 (-).
  Many rows have x1, x2, x4 near 0.
- Residuals show heteroscedasticity and heavy-ish tails (excess kurtosis ~0.75 under a Normal fit).

## Validation
5-fold CV (repeated with 2 different fold seeds), NLL computed on held-out folds, with
the whole fitting pipeline (standardisation, multiple random restarts, L-BFGS on penalised NLL)
re-run inside each fold.

| model | CV NLL |
|---|---|
| homoscedastic linear Normal, raw features | 3.417 (10-fold) |
| + engineered features (log age^2, water/binder, interactions) | 3.339 |
| MLP 6 tanh units, Normal, heteroscedastic (log-sd linear in hidden units) | 3.13-3.14 |
| same + water/binder input | no gain |
| MLP 6 units, Student-t, df 2.5 / 3.5 / 5 / 6 / 10 / 20 | 3.080 / 3.065 / 3.046 / 3.052 / 3.067 / 3.088 |
| MLP 7 units, Student-t df 6, L2 0.012 / 0.02 / 0.035 | 3.022 / 3.028 / 3.089 |
| MLP 7 units, Student-t df 4 / df 8 | 3.044 / 3.028 |

## Final program
- Inputs x0..x6 and log(x7) (x7 > 0 across the whole domain). The input standardisation
  is folded into the first-layer weights.
- 7 tanh hidden units. Location = linear in the hidden units; log-scale = linear in the hidden
  units, so the scale is bounded everywhere in the domain (tanh saturates). Training range of
  the scale is about 1.3 to 12.
- Student-t likelihood with fixed df = 6.
- Constants: 7*8 + 7 + 7+1 + 7+1 + 1 (df) = 80, which is exactly the budget. 445 AST nodes. Passes the contract check.
- Fit on all data: L2 penalty 0.015 (on standardised-input weights), best of 8 restarts.
  Training NLL is 2.889. The CV estimate of about 3.02 is the honest figure.

I chose it because it had the best CV NLL of the programs that fit the constant budget.
Seed-to-seed variation of the CV estimate is about ±0.03.

expected_nll: 3.03
