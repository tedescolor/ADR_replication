# Notes

## Data
1000 rows, x0 in (0,1). A scatter plot shows one noisy curve for x0 < ~0.47 that splits
into two branches (upper rises to ~2.5, lower falls to ~-1 at x0 = 1), with roughly
half the points on each branch. The noise sd drops from ~0.32 at x0 = 0 to ~0.10 at x0 = 1.
Residuals after a branch fit look Gaussian (pooled standardized skew 0.02, excess kurtosis -0.04).

## Validation
5-fold CV repeated twice (10 folds, different permutations), refitting by maximum likelihood
(BFGS + Nelder-Mead, several restarts) on each training split. Reported numbers are mean held-out NLL per point
(the standard error across folds is about 0.017, but the comparisons are paired, so differences are less noisy than that).

## Candidates (CV NLL)
- Two independent normal branches: polynomial means (degrees 6-10 / 3-4), log-linear or quadratic
  sds, sigmoid weight: 0.205-0.216. The fitted weight came out ~0.5 and constant, and both sds were about equal.
- Shared-structure model: centre c(x) = poly deg K in t = 2x-1, branches c -/+ h(x) with
  h = u*(b0*u+b1), u = max(0, x-k), k fitted (~0.475), common log-sd poly deg S, weight 0.5:
  - K=5,S=1: 0.205; K=6,S=1: 0.203; **K=6,S=2: 0.1965**; K=6,S=3: 0.197; K=7,S=2: 0.198;
    K=8,S=2: 0.200; K=5,S=2: 0.198; quadratic gap poly: 0.197; free weight: no gain.
- Sinusoidal centre instead of the polynomial: 0.198-0.209 (no better).

## Final
K=6, S=2 shared-structure mixture (13 fitted parameters, 18 literals in total). Train NLL 0.181,
CV NLL 0.1965. Fitting on all the data should be a little better than the CV fold fits,
but the result is noisy at this sample size, so I am using the CV value rounded slightly.

expected_nll: 0.197
