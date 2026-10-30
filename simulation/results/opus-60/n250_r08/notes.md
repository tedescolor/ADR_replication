# Notes

## Data
250 rows, x0 in [0,1]. Scatter shows one curve for x < ~0.55 (starts near 0, dips to
~-0.8 at x~0.2, rises through 0 at x~0.48). It then splits into two branches: an upper one
rising linearly to ~2.4 at x=1, and a lower one peaking at ~0.35 near x=0.6 and falling to ~-0.8
at x=1. Each branch holds about half of the points at high x. Noise is largest around x~0.2
(sd ~0.35) and smaller on the branches (sd ~0.15).

## Models (2-component Normal mixture, MLE fit with BFGS+Nelder-Mead and random restarts)
Validation: 10-fold CV repeated twice (different permutations), refitting every fold; the
score is the mean held-out NLL.
Config = (deg+1 of main mean, of main log-sd, of branch mean, of branch log-sd); gate = sigmoid(a+bx).

| config | #params | train NLL | CV NLL |
|---|---|---|---|
| (4,1,3,1) | 11 | 0.238 | 0.281 |
| (4,2,3,1) | 12 | 0.188 | **0.2355** |
| (4,2,3,2) | 13 | 0.184 | 0.2356 |
| (5,2,3,1) | 13 | 0.188 | 0.245 |
| (4,3,3,1) | 13 | 0.188 | 0.242 |
| (5,3,3,2) | 15 | 0.184 | 0.248 |
| (4,2,4,1) | 13 | 0.185 | 0.242 |

Capped gate w = sigmoid(-0.016)*sigmoid(k(x-c)) (13 params, 10-fold x1): train 0.168, CV 0.2346.
That ties (4,2,3,1) on CV but fits the picture better: the branch switches on near x~0.3 and
holds a ~50% share at high x. A Student-t main-component variant did not finish in time.

## Final
Capped-gate model: main component N(cubic(x), exp(linear(x))), branch component
N(quadratic(x), const), weight 0.496*sigmoid(16.45(x-0.296)). 13 fitted constants, refit on all
data (train NLL 0.168). CV NLL 0.2346. I nudged the estimate up a little for the selection bias of picking the best of 7.

expected_nll: 0.245
