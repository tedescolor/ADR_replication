# Notes

## Data
250 rows, x0 in (0,1). Scatter: a single noisy curve (dip to about -0.7 near x=0.2, rising after)
up to x≈0.6, where it splits into two branches. The upper branch is close to linear (≈0.7 → 2.5)
and the lower one is a concave parabola (≈0.2 → -0.8). After the split, about half the points are on each branch.
Noise is larger at small x and shrinks towards the split.

## Validation
5-fold CV repeated twice (10 fits per config), using mean held-out NLL. Optimisation is BFGS
from several starts (different split locations), warm-started from the full-data fit.

## Tried (train NLL / CV NLL)
- 2-component mixture, one smooth poly for main+upper branch, sigmoid-gated lower branch:
  deg4/lin lower 0.33/0.42; lower quadratic 0.29/0.385; Student-t, more scale terms, x-dependent
  gate weight: no gain.
- 3-piece model (pre-split curve | upper | lower, sharp sigmoid gate at learned c≈0.60,
  constant branch probability q): degrees (pre, upper, lower, logscale pre/up/low)
  (3,1,2,1,0,0) .292/.376; (3,1,2,1,1,1) .279/.371; (4,1,2,1,1,1) .278/.374;
  (5,1,2,1,1,1) .274/.361; **(5,1,2,1,0,1) .274/.359**; (6,1,2,1,1,1) .274/.368;
  quadratic upper or cubic lower or const pre-scale: worse.
- Pre-region-only check: symmetric bimodal noise vs normal noise. No consistent CV gain, so I kept normal noise.

## Final
Mixture of 3 Normals. Weights: [1-st, st(1-q), st q] with st = sigmoid(300(x-0.6002)), q≈0.476.
Pre-split mean is a quintic with log-linear scale. Upper branch is linear with constant scale. Lower branch is
quadratic with log-linear scale. 22 constants (209 AST nodes). Train NLL 0.274; CV NLL 0.359. I added a small
allowance for picking the best of about 20 configs.

expected_nll: 0.37
