# Notes

## Data
250 rows, x0 in [0,1], y in [-1.4, 2.7]. Binned summaries/scatter show a single S-shaped
curve for x < ~0.45 (min ~ -0.7 at x~0.25), which then splits into two branches: an upper
branch rising to ~2.5 at x=1 and a lower branch peaking ~0.3 near x~0.7 and falling to ~-0.95
at x=1. Roughly half the points on each branch. Noise is heteroscedastic: sd ~0.39 near x=0
decreasing to ~0.11 near x=1.

## Validation
5-fold CV repeated twice (10 fits per candidate, same folds for all candidates), multi-start
BFGS / Nelder-Mead MLE on each training fold, mean held-out NLL. All fits use t = 2*x0 - 1.

| candidate (2-component Normal mixture) | params | CV NLL |
|---|---|---|
| separate cubic means, linear log-sd per comp, logistic-linear weight | 14 | 0.249 (1x5 fold) |
| deg 4/5/6 means, more sd/weight terms | 15-22 | 0.25-0.34 (overfit) |
| separate cubic means, shared linear log-sd, const weight | 11 | 0.192 |
| same + logistic weight | 12 | 0.194 |
| same, separate sds | 13 | 0.205 |
| same, quadratic log-sd | 12 | 0.208 |
| same, constant sd | 10 | 0.258 |
| quadratic/cubic or quartic means | 10-12 | 0.20-0.30 |
| Student-t components (shared df) | 12 | 0.192 |
| shared cubic g(t) + per-branch hinge a_k r^2 + b_k r^3, r=max(t-t0,0) | 12 | 0.186 |
| shared cubic + hinge r^3 only w/o g cubic term | 11 | 0.179 |
| **shared cubic g(t) + per-branch hinge a_k r^2** | 10 | **0.176** (se ~0.03) |

## Final program
y | x ~ w N(g(t) - a1 r^2, s(t)) + (1-w) N(g(t) - a2 r^2, s(t)),
g cubic in t, r = max(t - t0, 0) (branch point x ~ 0.40), s = exp(linear in t), w constant
(~0.48). 10 fitted parameters, 14 literals in the program. Fitted by MLE on all 250 rows
(in-sample NLL 0.142). Chosen because it encodes the observed structure (one curve that
bifurcates) with the fewest parameters and had the best CV score.

## Estimate
CV NLL 0.176 with training sets of 200; the full-data fit should be a bit better, but
selection among ~12 candidates on the same folds adds optimism, so I take ~0.18.

expected_nll: 0.18
