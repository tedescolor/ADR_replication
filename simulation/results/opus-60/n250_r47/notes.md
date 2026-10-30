# Notes

## Data
250 rows, x0 in (0, 1). Scatter plot: for x0 < ~0.6 there is one noisy, wavy trunk
(sd ~0.3); around x0 ~ 0.6 it splits into two near-linear branches, an upper one rising to ~2.5
and a lower one falling to ~-0.9 at x0 = 1. The branches are tighter than the trunk (sd ~0.07-0.1) and
each holds about half the points at every x0 above 0.65.

## Validation
Repeated K-fold CV of the mean held-out NLL: 5 folds × 2–4 repeats, refitting by maximum likelihood
(BFGS, several starts) on each training split.

| model | #params | train NLL | CV NLL |
|---|---|---|---|
| single Gaussian, poly mean + log-linear sd | 6 | 0.79 | 0.83 |
| 2-comp Gaussian mixture, quadratic means, log-lin sd, logistic weight | 12 | 0.37 | 0.43 |
| trunk cubic + softplus bifurcation, constant sds | 11 | 0.307 | 0.366 |
| same, quintic trunk, const sd | 13 | 0.306 | 0.382 |
| trunk cubic + bifurcation, log-linear sd per branch | 13 | 0.184 | 0.245–0.248 |
| same, quartic / quintic trunk | 14/15 | 0.181–0.183 | 0.248–0.255 |
| same + sine term in trunk | 16 | 0.176 | 0.266 (overfits) |
| log-quadratic sds | 15 | 0.177 | 0.248 |
| **gated sd (trunk sd → branch sds, sigmoid gate), const weight** | 13 | 0.184 | **0.244** |
| gated, trunk-linear log-sd | 14 | 0.180 | 0.242 |
| gated, branch-linear log-sd | 15 | 0.177 | 0.246 |

The last few differ by less than the CV noise (~0.005). I chose the 13-parameter gated model
because it ties for best and is simple to interpret.
The empirical branch fractions are about 0.5 across x, so a constant weight is enough.

## Final program
mean trunk f(x) = cubic; h(x) = s·softplus((x−c)/s), with c = 0.613 and s = 0.075;
upper = f − 1.95h, lower = f − 11.0h (the cubic itself rises, so the "upper" branch is f plus a small
correction). log sd = (1−g)·(−0.76) + g·(branch level), with g = sigmoid((x−c)/0.27);
mixture weights 0.525/0.475. The program has 20 literal constants and 176 AST nodes. Train NLL = 0.184.

## Estimate
CV NLL = 0.244 (the CV fits were warm-started from the full-data fit, which in an earlier comparison
made CV about 0.003 too optimistic). Allowing for that and for selecting among several models:

expected_nll: 0.25
