# Notes

## Data
250 rows, x0 in (0,1). Scatter shows one noisy curve for x < ~0.55 (dip near x=0.25,
upturn at x=0), which splits into two branches: an upper branch rising
roughly linearly to y≈2.5 at x=1 and a lower branch peaking near 0.2 at x≈0.7 and falling
to ≈ -1 at x=1. Noise shrinks with x (sd ≈0.4 at x=0, ≈0.1 at x=1).

## Model family
Two-component Gaussian mixture, fitted by maximum likelihood (BFGS + Nelder-Mead):
- main component: mean = polynomial of degree dm in x, log-sd linear in x;
- branch component: mean = polynomial of degree db, log-sd linear;
- weight of branch = sigmoid(k (x - c)).

## Validation
5-fold CV repeated 3 times (different permutations), refitting on each training split
starting from the full-data fit. Mean held-out NLL per point:

| spec (dm,db,ds1,ds2) | params | train | CV |
|---|---|---|---|
| 3,1,1,0 | 11 | 0.224 | 0.299 |
| 4,2,1,1 | 14 | 0.122 | 0.203 |
| **5,2,1,1** | 15 | 0.110 | **0.189** (se≈0.044) |
| 6,2,1,1 | 16 | 0.109 | 0.191 |
| 5,2,2,1 | 16 | 0.110 | 0.193 |
| 5,2,1,2 | 16 | 0.109 | 0.206 |
| 5,3,1,1 | 16 | 0.110 | 0.201 |
| 5,2,1,1 + Student-t main | 16 | 0.110 | 0.194 |

Higher degrees, quadratic log-sd and heavy tails did not help; (5,2,1,1) has the best CV and
uses 15 fitted constants (17 literals total).

## Final
(5,2,1,1) refit on all 250 rows. Train NLL 0.110; CV 0.189. Allowing a little for
optimism of selecting the best of several specs:

expected_nll: 0.20
