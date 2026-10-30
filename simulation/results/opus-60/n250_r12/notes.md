# Notes

## Data
250 rows, x0 in [0,1]. Scatter: a single noisy curve for x < ~0.5 (dip to -0.9 near x=0.25,
residual sd ~0.3), which forks at x ~ 0.5 into an upper branch (rising to ~2.5 at x=1) and a
lower branch (peaks near 0.65 and falls to ~-0.9 at x=1). The noise shrinks as x grows.

## Validation
10-fold CV repeated twice (20 folds, 2 different shuffles); mean held-out NLL per row. Every
candidate is refit from scratch on each training fold (MLE, BFGS, several random starts, tiny
L2 penalty of 1e-3).

## Candidates (CV NLL)
- 2-component normal mixture, cubic means, log-linear sds, logistic-linear weight: 0.264-0.270
- same with quadratic log-sd: 0.281; quartic means: 0.273; constant sds: 0.280
- hinge model: shared cubic base, each branch adds terms in h=max(x-c,0) with c fitted,
  log-linear sds, logistic weight: 0.259
- reduced hinge (upper component = base with no hinge term, constant weight): **0.239**
  (training NLL 0.185, so about the same fit with fewer free parameters, and fits that are stable across folds)

## Final program
Mixture of two Normals with weight 0.49/0.51. Comp a: mean = cubic base(x), sd = exp(-0.89-1.28x).
Comp b: mean = base(x) - 3.77 h - 6.34 h^2, h = max(x-0.4948, 0), sd = exp(-0.83-1.36x).
13 fitted parameters (about 16 literal constants). Full-data training NLL is 0.182.
The estimate below is the CV value, rounded up a little because breakpoint fitting adds variance.

expected_nll: 0.25
