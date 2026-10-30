# Notes

## Data
250 rows, x0 in [0,1]. For x < ~0.55 y follows one noisy curve (dip near x=0.25, sd ~0.3).
Above ~0.6 the data split into two branches: an upper, almost straight line (y ≈ 4.8x − 2.2, sd ~0.1)
and a lower branch that keeps the oscillating curve and falls to about −1 at x=1. At high x the
two branches hold roughly equal shares of the points.

## Model family
Two-component Gaussian mixture with an x-dependent weight:
- main: mean a·sin(b x + c) + d + e x, log-sd linear in x;
- upper: mean linear in x, log-sd linear in x;
- weight of upper = sigmoid(linear in x).

## Validation
Repeated K-fold CV: 5 folds × 3 random repeats. Each training fold was refit by BFGS MLE,
started from the full-data fit plus jitter. I report the mean held-out NLL per point.

| model | params | train NLL | CV NLL |
|---|---|---|---|
| poly5 main mean, linear log-sd, linear upper (const sd) | 13 | 0.264 | 0.315 |
| poly5 main, linear log-sds (both) | 14 | 0.253 | 0.313–0.320 |
| poly6 main | 15 | 0.250 | 0.331 |
| poly4 main, quadratic log-sd, quadratic logit weight | 15 | 0.227 | 0.309 (one run; fragile) |
| sin main (no linear term) | 12 | 0.286 | 0.341 |
| **sin + linear main (final)** | 13 | 0.251 | **0.3145** |
| + Student-t main | 14 | 0.251 | 0.3145 (df → ∞) |
| + quadratic upper mean | 14 | 0.251 | 0.324 |
| + quadratic log-sd main | 14 | 0.251 | 0.319 |
| + skew-normal main | 14 | 0.251 | 0.318 |
| + 3rd broad component | 15 | 0.223 | 0.299, rejected: collapses to a spike (sd≈e^-21) on one point |

I chose the sin+linear model because it is among the best by CV and has few parameters,
and its fits are stable. None of the extensions improved CV.

## Final
13 constants. Train NLL 0.251, CV NLL 0.3145. The final model is trained on 100% of the data
rather than the 80% used in each fold, so the true NLL may be slightly lower. Small-sample
noise in the held-out set can move it either way.

expected_nll: 0.31
