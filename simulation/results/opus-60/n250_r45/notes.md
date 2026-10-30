# Notes

## Data
250 rows, x0 in (0,1). Scatter shows one noisy curve for x < ~0.5 that splits into two
branches: an upper roughly linear branch (y ~ -2.3 + 4.9x, reaching 2.5 at x=1) and a lower
wavy branch (min ~-0.85 at x~0.25, max ~0.15 at x~0.7, down to -1 at x=1). Noise shrinks
with x (sd ~0.4 near x=0, ~0.1 near x=1).

## Model family
Two-component Gaussian mixture, all parameters functions of x:
mean1 = quartic poly, mean2 = linear, log sd1 / log sd2 = linear, mixing logit = quadratic.
Fitted by maximum likelihood (BFGS, several inits).

## Validation
Repeated K-fold CV (refit on every fold), mean held-out NLL:
| spec (deg m1,m2,logs1,logs2,logit) | train | CV |
|---|---|---|
| (3,1,0,0,1) const sd | 0.328 | 0.371 |
| (3,1,1,1,1) | 0.220 | 0.266 |
| (4,1,1,1,1) | 0.206 | 0.257 (5f) / 0.262 (10f) |
| **(4,1,1,1,2)** | 0.181 | 0.243 (5f x2), 0.252 (5f x3), 0.247 (10f x2) |
| (4,1,1,1,3) | 0.175 | 0.253 |
| (5,1,1,1,2) | 0.181 | 0.256 |
| (6,1,1,1,2) | 0.166 | 0.246 |
| (4,2,1,1,2) | 0.181 | 0.256 |
| (4,1,2,1,2) | 0.181 | 0.262 |
| (4,1,2,2,2) | 0.179 | 0.309 |
| Student-t lower comp. | 0.206 | 0.259 (no gain, df -> large) |

The chosen spec is the smallest one at the CV optimum; more flexible variants do not
improve held-out NLL. Final model refit on all data: 14 constants, train NLL 0.181.

## Expected performance
CV estimates 0.243-0.252; I take ~0.25.

expected_nll: 0.25
