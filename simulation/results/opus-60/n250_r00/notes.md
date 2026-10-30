# Notes

## Data
250 rows, x0 in (0,1). Scatter shows a single noisy curve (min ~ -0.75 near x=0.2) up to
x ~ 0.47, where it splits into two branches: an upper one rising almost linearly to ~2.4 at
x=1, and a lower one bending down to ~ -1 at x=1 (about 40/60 lower/upper). Noise shrinks
with x (sd ~0.4 near x=0, ~0.25 mid-range, ~0.1 on the branches near x=1).

## Validation
10-fold CV repeated twice (20 fits per candidate), mean held-out NLL per point. Each fold
fit uses BFGS MLE from several generic inits; I also add the full-data optimum as one start
because the hinge/mixture likelihood has bad local optima that otherwise made CV noisy.
(Only the start point is shared; the fold parameters are fit on the training folds only.)

## Candidates (CV NLL / train NLL)
- free 2-comp Gaussian mixture, polynomial means (deg 3-5), log-poly sds, logistic weight:
  0.219-0.26 (overfits more as the degree goes up)
- trunk cubic + hinge at c, quadratic branch offsets, separate log-linear sds: 0.2035 / 0.150
- same with shared log-linear sd: 0.1996; shared log-quadratic sd: 0.202
- same with shared sd = |a(1-x)| + b: **0.1937 / 0.1456** (chosen)
- linear (not quadratic) branch offsets: 0.32; Student-t components: no gain (nu -> 15);
  x-dependent mixture weight: 0.197-0.207 (no gain)
One point lying between the branches (held-out NLL ~4.5) appears in every model's CV.

## Final program
Mixture of 2 Normals. Trunk f(x) is cubic; h = max(x - 0.4687, 0); each branch is
f + a1*h + a2*h^2; shared sd 0.2775*(1-x) + 0.0853; constant weight sigmoid(-0.3685).
17 constants. Train NLL 0.1456, CV NLL 0.1937. I raised the estimate slightly to allow
for having picked the best of about 10 candidates.

expected_nll: 0.20
