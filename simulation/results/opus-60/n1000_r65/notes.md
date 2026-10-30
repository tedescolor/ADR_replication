# Notes

## Data
1000 rows, x0 in [0,1]. Scatter shows two branches for x > ~0.45: a rising, nearly linear branch
(y ~0 at x=0.5 to ~2.5 at x=1) and a falling wiggly branch (to ~-1 at x=1), both narrowing as x grows.
For x < ~0.45 there is one wider noisy cloud (sd ~0.3) dipping to about -0.8 near x=0.2.

## Model family
A 2-component Gaussian mixture. Each component mean is a polynomial in t = 2x-1, each log-sd is a
polynomial, and the weight is sigmoid(polynomial). Fit by direct maximum likelihood (L-BFGS, Chebyshev
basis, 4 restarts from branch-shaped initial values), then converted to power-basis Horner form.

## Validation
5-fold CV (fixed permutation, seed 1), refitting from scratch in every fold. Degrees are (mean, log-sd, weight):

| degrees | train | CV NLL |
|---|---|---|
| 3 1 1 | 0.213 | 0.2277 |
| 4 1 1 | 0.209 | **0.2269** |
| 4 2 2 | 0.203 | 0.2270 |
| 5 1 1 | 0.207 | 0.2272 |
| 5 2 1 | 0.203 | 0.2308 |
| 5 1 2 | 0.207 | 0.2301 |
| 5 2 2 | 0.202 | 0.2300 |
| 6 2 2 | 0.200 | 0.2404 |
| 6 3 2 | 0.198 | 0.2349 |
| 7 3 3 | 0.196 | 0.2419 |

Higher degrees overfit. 3/4/5 with linear scale/weight are tied within noise, so I chose the
smallest competitive model, (4,1,1): 16 fitted constants (18 literals in total, counting the 2 in t=2x-1 and the 1 in 1-w).
The weight ends up nearly constant (about 0.52). The probability integral transform (PIT) of the final fit,
plotted against x, looks uniform, so the calibration looks fine. The final fit on all data has train NLL 0.2086.

## Estimate
The CV NLL is 0.227, and per-fold values range from 0.16 to 0.27. I added a small allowance for
the optimism of picking the best model among several.

expected_nll: 0.232
