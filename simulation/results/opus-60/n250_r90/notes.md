# Notes

## Data
250 rows, x0 in [0,1]. Scatter shows one noisy curve for x0 < ~0.55 (dips to ~-0.7 near
x0 = 0.2 and rises after that). At about x0 = 0.55 it splits into two branches of roughly equal
mass: an upper, nearly linear branch (y ~ -2.3 + 4.9 x0, getting tighter as x0 grows) and a
lower branch that bends down to about -0.9 at x0 = 1. Noise in the main curve shrinks as x0 grows
(sd ~0.35 at small x0, ~0.12 at large x0).

## Validation
5-fold CV, repeated with 2 random partitions (10 fits per candidate), refitting by MLE
(BFGS) on each training split. Mean held-out NLL is reported.

## Candidates (CV NLL)
- 2-component Normal mixture with polynomial means/log-scales and a logit-linear weight:
  0.25-0.30 (cubic main mean was clearly worse: 0.30)
- Weight replaced by w_max * sigmoid(k (x - c)) as a gate: 0.225; the fitted k was very large
  and w_max ~ 0.5
- Fixing k = 100 (soft but sharp gate, more robust near the split): 0.222
- Also fixing w_max = 0.5: 0.215
- Adding a quadratic log-scale, a quadratic upper-branch mean, or a quintic main mean: no gain (0.218-0.230)
- Student-t main component (df fitted to ~8.7): 0.214, the best. I kept it also because heavier
  tails make it more robust to outliers.

## Final program
Mixture of (1-w) StudentT(8.7, quartic(x), exp(a+b x)) and w Normal(linear(x), exp(c+d x)),
with w = 0.5 * sigmoid(100 (x - 0.546)). 15 constants in total, refit on all data.
Train NLL 0.156; CV NLL 0.214 (standard error of the mean ~0.05).

expected_nll: 0.22
