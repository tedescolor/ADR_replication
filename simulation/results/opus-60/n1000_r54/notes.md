# Notes

## Data
1000 rows, x0 uniform on [0,1]. Scatter plot: a single noisy curve (sd ~0.35 -> 0.27) for x < ~0.53;
beyond that y splits into two roughly equal-weight branches: a rising, nearly linear one (to ~2.45 at x=1)
and a falling one (to ~-1 at x=1). Both branches get tighter as x increases (sd ~0.1).

## Model
Two-component Normal mixture, z = x - 0.5:
- component A (main curve + lower branch): mean = degree-7 polynomial, log sd = quadratic;
- component B (upper branch): mean = quadratic, log sd = quadratic;
- weight of B = sigmoid(k (z - c)) * sigmoid(w): an almost step-like switch at x ~0.529, max weight ~0.48.
20 fitted constants (+ the 0.5 centring), fitted by maximum likelihood (BFGS).

## Validation
5-fold CV (random split, seed 0), refitting each fold from the full-data fit.
| mean deg / branch deg / log-sd deg | weight | params | train NLL | CV NLL |
|---|---|---|---|---|
| 5/2/2 | max fixed 0.5 | 17 | 0.217 | 0.229 |
| 7/2/2 | max fixed 0.5 | 19 | 0.211 | 0.228 |
| 9/3/2 | max fixed 0.5 | 22 | 0.209 | 0.231 |
| 9/3/3 | max fixed 0.5 | 24 | 0.202 | 0.231 |
| 6/2/2 | free max | 19 | 0.212 | 0.229 |
| **7/2/2** | **free max** | **20** | **0.210** | **0.227** |
| 8/2/2 | free max | 21 | 0.210 | 0.227 |
| 7/2/1 | free max | 18 | 0.220 | 0.237 |
| 7/2/2, Student-t component A | free max | 21 | 0.210 | 0.227 (no gain) |

Chose 7/2/2 with free maximum weight: best CV, and larger models don't improve.
Heavy tails didn't help, so I kept Normal components. Final model NLL on data.csv: 0.2100.
I put the expected held-out NLL slightly above the CV value to allow for selection optimism.

expected_nll: 0.23
