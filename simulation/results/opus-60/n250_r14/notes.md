# Notes

## Data
250 rows, x0 in (0,1), y in (-1.4, 2.5). A scatter plot shows two crossing curves:
- a main branch, smooth and rising from ~0 (x=0) via ~-0.75 (x=0.2) to ~2.5 (x=1), with noise
  sd falling from ~0.4 at x=0 to ~0.1 at x=1;
- a second branch, roughly a parabola (≈-1.25 at x=0.22, peak ≈0.1 at x≈0.65, ≈-0.85 at x=1),
  sd ~0.2, holding ~40% of points, and **missing for x0 < ~0.22** (no points down at -2..-3 there).
Above x≈0.6 y is clearly bimodal.

## Validation
10-fold CV repeated 3 times (30 fits per model), mean held-out NLL per point; SE ≈ 0.05
(across points, so differences between models are much better determined than that). Each fit
is BFGS MLE with random restarts. Fold fits start from the full-data optimum (plus restarts),
which may make CV slightly optimistic for models with several local optima.

## Results (CV NLL, #params)
| model | k | CV |
|---|---|---|
| mixture, poly means (4,2), log-linear sds, logistic weight in x | 14 | 0.232 |
| same, Student-t lower comp. (df -> inf) | 15 | 0.230 |
| same, sinusoidal lower mean | 14 | 0.234 |
| same, quintic lower mean | 15 | 0.229 |
| same, quadratic log-sd | 15 | 0.224 |
| logistic weight, crossing-branch init | 13 | 0.233 |
| **gated second branch** w = p·sigmoid(k(x-c)), k free (~5000) | 15 | 0.195–0.197 |
| gated, k fixed at 15 / 40 / 100 / 1000 | 13 | 0.198 / 0.195 / 0.192 / 0.192 |
| gated, constant sd for second branch | 12–13 | 0.192–0.200 |
| gated, both sds constant | 12 | 0.261 |

The key structural point is the gate: the second branch only exists for x0 > ~0.22. A smooth
logistic weight cannot switch it off fast enough. Everything inside the gated family is within
noise (0.192–0.200), so I picked the 13-constant version with a fixed gate slope of 100
(transition width ~0.04). It is as good as the hard step and less brittle if the true onset
is not exactly at 0.221.

## Final program
Mixture of 2 Normals:
- A: cubic mean, log-sd linear in x0;
- B: quadratic mean, log-sd linear in x0;
- weight of B = sigmoid(-0.456) · sigmoid(100·(x0 − 0.221)).
13 fitted constants (plus the fixed 100). Train NLL 0.135, CV NLL 0.192.

## Expected NLL
The CV estimate is 0.192. I add a margin for picking the best of ~15 candidates, for the
optimistic warm-started folds, and for doubt about where the gate sits (some fits put it at
0.375 instead).

expected_nll: 0.21
