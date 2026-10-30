# Notes

## Data
1000 rows, x0 roughly uniform on [0,1]. Scatter plot: for x < ~0.5 there is one noisy curve
(sd ~0.3, falling to ~0.22 near 0.5) that dips to about -0.8 at x≈0.2. Near x≈0.51 it splits into
two branches with roughly equal weight. The upper branch keeps rising to ~2.5 at x=1 and the
lower one bends down to ~-1. Each branch has sd shrinking to ~0.1 at x=1. Local 2-component
EM fits in x-windows confirmed this (weights ≈0.5/0.5 after the split).

## Validation
5-fold CV repeated twice (10 folds, fixed seed). Every candidate is refit on each training fold
with the same multi-start BFGS/Nelder-Mead procedure, and I report the mean held-out NLL ± SE.
Note: one early exploratory batch ran 4 processes at once by mistake; everything after that
used ≤2.

## Candidates (CV NLL)
- 2-comp Gaussian mixture, polynomial means/log-sds in u=2x-1, logistic gate: degrees 3/1: 0.230,
  higher degrees 0.242–0.251 (overfit and local optima).
- Sinusoidal lower branch + polynomial upper branch: 0.256–0.279 (worse).
- **Split model**: constant mixture weight. Upper mean m1 = cubic in u. Lower mean
  m2 = m1 − (a r + b r²), with r = softplus(40(x−c))/40, so the branches coincide before the
  split point c. Each component has a quadratic log-sd.
  - mean deg 3, sd deg 2, split quadratic: **0.2267 ± 0.012** (14 params) ← chosen
  - mean deg 4: 0.2288; deg 5: 0.2295–0.2308; sd deg 1: 0.2296–0.2303; sd deg 3: 0.2296;
    linear split: 0.242; ReLU instead of softplus: 0.229; x-dependent gate: 0.230.

I chose the smallest model with the best CV. PIT values are uniform overall and within every
x-decile, so I found no leftover tail or heteroscedasticity problems.

## Final program
A Mixture of 2 Normals, refit on all data. It has 14 fitted parameters and about 22 literal
constants. Training NLL is 0.2089. CV NLL (fits on 80% of the data) is 0.2267. The final fit uses
100% of the data, so it should do slightly better than the CV figure.

expected_nll: 0.226
