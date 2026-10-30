# Notes

## Data
1000 rows, x0 in [0,1], y in [-1.6, 2.7]. Scatter: for x < ~0.5 a single noisy curve
(dips to -0.85 near x=0.2, rises to ~0.4 at x=0.6, noise sd ~0.3 shrinking with x).
From x ~0.5 it splits into two branches of roughly equal mass: an upper branch rising to
~2.5 at x=1 and a lower branch that stays near 0 until x~0.8 then falls to ~-0.9.
Noise shrinks to ~0.1 at x=1 on both branches.

## Model family
2-component Gaussian mixture:
- m1(x): degree-5 polynomial (upper/main branch, also the single curve on the left);
- m2(x) = m1(x) + d(x), with d = 0 for x < k (a hinge), so the components coincide on the left;
- log sd of each component: quadratic in x;
- mixture weight: logistic of a polynomial in x.

## Validation
5-fold CV (fixed random split, seed 0), refitting all parameters by ML
(BFGS + Nelder-Mead, several restarts) on each training split. Variants (CV NLL):
- d = h*cubic(h), weight logistic-linear: 0.2202 (train 0.1999)
- d = h*quartic(h): 0.2180
- d = -c*h^p (power hinge): 0.2190
- extra constant/linear offset between components on the left (to capture slightly
  platykurtic left-region noise, gennorm beta ~2.7 there): 0.2206 / 0.2195 (no gain)
- deg-6 mean, cubic log-sd: 0.2201 / 0.2200
- constant weight: 0.2189; quadratic logit weight: 0.2208
- **power hinge + constant weight (final): 0.2169** (train 0.2013), 16 constants
- same with linear log-sd: 0.2191
Differences between variants are mostly within CV noise (per-fold values range
0.17-0.30); I chose the best-CV variant, which is also among the most parsimonious.

## Final program
w = sigmoid(0.065) (~0.52); m1 quintic; d = -exp(2.235) * max(x-0.479,0)^exp(0.414)
(i.e. ~ -9.3 h^1.51); log s1, log s2 quadratic. 16 fitted constants.
Fitted on all 1000 rows. Train NLL 0.2013, CV NLL 0.2169. Adding a bit for
selection among ~12 variants:

expected_nll: 0.225
