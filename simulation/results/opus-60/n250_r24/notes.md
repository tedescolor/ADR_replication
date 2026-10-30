# Notes

## Data
250 rows, x0 in [0,1], y in [-1.3, 2.6]. For x < ~0.5 y follows one noisy curve
(dip to ~-0.85 near x=0.2, back to 0 at x=0.5). Above x~0.55 the data split into two
clear branches: an upper one rising to ~2.5 at x=1 and a lower one falling to ~-0.9.
Noise sd shrinks with x (~0.35 at x=0, ~0.12 at x=1).

## Model family
Two-component Gaussian mixture; branch means are polynomials in x, log-sd polynomial in x
(shared or separate), mixing weight logistic-polynomial in x (or constant / fixed 0.5).
Fitted by MLE (BFGS, several restarts) — scripts in work/fit.py, work/fit2.py.

## Validation
10-fold CV repeated twice (different permutations), refitting from scratch in each fold;
mean held-out NLL over all 500 held-out predictions (se ~0.05, but differences between
specs are paired and much less noisy).

| means (deg) | log-sd | weight | k | train | CV |
|---|---|---|---|---|---|
| 2,2 separate sd lin | sep | lin | 12 | 0.355 | 0.434 |
| 3,3 | sep lin | lin | 14 | 0.159 | 0.232 |
| 4,4 | sep lin | lin | 16 | 0.149 | 0.253 |
| 3,3 | sep const | lin | 12 | 0.230 | 0.315 |
| 3,3 | shared lin | lin | 12 | 0.166 | 0.220 |
| 3,3 | shared lin | const | 11 | 0.172 | 0.222 |
| 3,3 | shared lin | 0.5 | 10 | 0.181 | 0.224 |
| 3,3 | shared quad | const | 12 | 0.166 | 0.227 |
| 3,3 Student-t | shared lin | const | 12 | 0.172 | 0.222 |
| **3,4** | **shared lin** | **const** | **12** | **0.158** | **0.210** |
| 3,4 | shared lin | 0.5 | 11 | 0.162 | 0.214 |
| 4,4 | shared lin | const | 13 | 0.156 | 0.209 |
| 5,5 | shared lin | const | 15 | 0.156 | 0.221 |
| 3,2 | shared lin | const | 10 | 0.771 | 0.747 |

## Choice
Upper branch cubic, lower branch quartic, shared log-linear sd, constant weight (~0.57
upper). Best CV among the simpler specs (4,4 is equal within noise with one more parameter).
Student-t tails and x-dependent weights gave no gain. Final program refit on all 250 rows:
12 fitted constants (13 literals incl. the 1.0 in 1-w); train NLL 0.158.

CV of the chosen spec is 0.210; allowing a little for selection among ~15 specs, my
estimate for new data is ~0.22.

expected_nll: 0.22
