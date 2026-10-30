# Notes

## Data
1000 rows, x0 in (0,1), y in about (-1.6, 2.6). A scatter plot shows one noisy curve for
x < ~0.5 (it dips to about -0.8 near x = 0.25 and rises to ~0.1 near 0.55). Around x ~ 0.5
it **splits into two thin branches**: one rises almost linearly to ~2.5 at x = 1, the
other bends down to ~-1. The noise sd shrinks from ~0.45 at x = 0 to ~0.1 at x = 1.
Residuals within a branch look Gaussian (skew/kurtosis near 0). The fraction of points on
each branch is ~0.5 at every x past the split.

## Model family
2-component Gaussian mixture: both means = base curve f(x) + branch offset
a*r + b*r^2, with r = max(x - k, 0), so the two components are identical before the knot k.
Shared noise sd exp(poly(x)); mixture weight constant.

## Validation
2 x 5-fold CV (same folds for every candidate, so comparisons are paired), multi-start
BFGS + Nelder-Mead refit in each fold. Mean held-out NLL:

| variant | #params | CV NLL |
|---|---|---|
| generic polys, separate sd per comp, logistic weight (first scan, other folds) | 20-25 | 0.232-0.240 |
| A: poly5 f, quad log-sd, free weight | 15 | 0.2253 |
| A, weight fixed 0.5 | 14 | 0.2249 |
| A, poly6 f | 16 | 0.2262 |
| A, cubic branch offsets | 17 | 0.2263 |
| A, softplus (smooth) knot | 16 | 0.2258 |
| A, linear log-sd | 14 | 0.2286 |
| A, cubic log-sd | 16 | 0.2239 |
| A, cubic log-sd, w = 0.5 | 15 | 0.2232 |
| B: f = c0 + c1 x + c2 sin(c3 x + c4), quad log-sd, w = 0.5 | 13 | 0.2243 |
| **B, cubic log-sd, w = 0.5 (final)** | 14 | **0.2228** |
| lower branch = f, upper linear only (sine f) | 12 | 0.2467 |

The differences between the top variants are small (~0.002, paired). I chose the final
model because it has the best CV score, few parameters, and a smooth base curve. Fitted
on all the data, its training NLL is 0.2073.

## Final program
f = 0.81156 - 0.71474 t + 1.45290 sin(4.47060 t + 3.76436); knot at 0.49733;
up = f - 1.762 r + 7.722 r^2, down = f - 5.260 r + 0.893 r^2; log-sd cubic in t; equal
weights. That is 14 fitted parameters (17 literals counting 0.0 and 0.5/0.5).

## Expected NLL
The CV estimate is 0.2228. The full-data fit should be slightly better than the 80% fold
fits, but choosing the best of ~12 variants adds some optimism, so I estimate ~0.225.

expected_nll: 0.225
