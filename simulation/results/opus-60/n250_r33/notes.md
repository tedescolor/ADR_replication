# Notes

## Data
250 rows, x0 in [0,1]. The scatter plot shows one smooth curve for x0 < ~0.56: it starts near 0, dips to
about -0.75 at x0 ~ 0.2 and rises to about 0.25 at x0 ~ 0.55. Past that point y splits into two branches:
- a lower branch that continues the curve and bends down to about -0.9 at x0 = 1;
- an upper, roughly linear branch rising from about 0.5 to 2.5.
Each branch holds roughly half of the points. Noise is small (sd ~0.1-0.25).

## Model family
Two-component mixture. Lower component: polynomial mean with a log-linear scale in x0. Upper component:
polynomial mean with a log-linear or constant scale. The upper component's weight is
pmax * sigmoid(k (x0 - c)). Fitting is by maximum likelihood (BFGS).
- Variants compared: lower degree 3/4/5; upper degree 1/2; gate steepness k in {15, 30, 40, 60} (fixed,
  because a free k ran off to a hard step); scale slopes on or off; Normal vs Student-t components.

## Validation
10-fold CV repeated twice (500 held-out predictions); the standard error of the mean NLL is ~0.065.
Selected results (CV mean NLL):
- Normal, deg 4/1, both scale slopes: 0.161. Deg 4/2: 0.148. Deg 4/2 with constant upper scale: 0.137.
- Student-t, deg 4/1, constant upper scale: 0.133 (k=30), 0.127 (k=40), 0.129 (k=60).
- Student-t, deg 3/1: 0.153. Deg 5/1: 0.145.

Most of the CV loss comes from a few points: the upper-branch point at x0 = 0.994 (extrapolation),
(0.623, 1.43) and (0.773, -0.32). Student-t tails cut the worst losses. All the differences are within
about 1 SE, so I chose the simplest robust variant that was also among the best.

## Final program
Student-t mixture: quartic lower mean, log-linear lower scale, df ~10.8. Linear upper mean, constant
upper scale 0.13, df ~4.7. Gate: 0.474 * sigmoid(40 (x0 - 0.561)). Fitted on all the data: 16 constants.
Training NLL is 0.056 and CV NLL is 0.127. Fitting on the full data should be slightly better than the
CV fits, but the tails and edges add risk, so my estimate is a bit above the CV value.

expected_nll: 0.14
