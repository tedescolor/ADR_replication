# Notes

## Data
250 rows, x0 in [0,1]. The scatter shows two branches. A lower curved branch (y ~ -0.1 at x=0,
min ~ -0.7 near x=0.2, ~0 at x=0.7, -0.9 at x=1) has a spread that shrinks with x (sd ~0.35 down to ~0.1).
An upper, almost linear steep branch (slope ~5, y ~0.1 at x=0.5 to 2.6 at x=1) holds about half the
points once it is present. The two branches overlap for x in 0.3-0.55, and there are no
points where the upper line would sit for x < 0.28.

## Model family
2-component Normal mixture. Upper weight = c * sigmoid((x - t)/s). Lower mean is a polynomial in
z = x - 0.5, the upper mean is linear, and both log-sds are linear in z. Fitted by maximum likelihood
(BFGS, multi-start).

## Validation
5-fold CV repeated (4 reps for the first candidate, then 2 reps with folds warm-started from the full-data fit;
this is slightly optimistic). Per-point NLL standard error ~0.05, but all candidates share the same folds, so
differences between candidates are less noisy than that.

| lower deg | upper deg | lower log-sd deg | upper log-sd deg | gate | CV NLL |
|---|---|---|---|---|---|
| 4 | 1 | 1 | 0 | free (sharp) | 0.199 |
| 4 | 1 | 1 | 0 | width >= 0.03 | 0.209 |
| 4 | 1 | 1 | 0 | width >= 0.08 | 0.212 |
| 3 | 1 | 1 | 0 | free | 0.198 |
| 5 | 1 | 1 | 0 | free | 0.198 |
| 4 | 2 | 1 | 0 | free | 0.202 |
| 4 | 1 | 2 | 0 | free | 0.198 |
| 4 | 1 | 1 | 1 | free | 0.185 |
| 4 | 1 | 2 | 1 | free | 0.188 |
| **3** | **1** | **1** | **1** | **free** | **0.182** |

A heteroscedastic upper branch is the only clear gain. A sharp gate (at x ~ 0.28, the point where the
upper line drops below the data) beats smooth gates. The final model is the simplest of the best
group: cubic lower mean, linear upper mean, log-linear sds, and 15 constants. Its NLL on the training data is 0.132.

## Expected NLL
The CV estimate is 0.182. I add a margin because the warm-started CV is optimistic and the sharp gate
location is uncertain.

expected_nll: 0.20
