# Notes

## Data
1000 rows, x0 in (0,1). Scatter shows one curve for x < ~0.51 (dips to -0.75 near
x=0.2, then rises roughly linearly to ~2.45 at x=1) and, from x ~0.51, a second branch
splitting off and bending down to ~-0.95 at x=1. About half of the points sit on each
branch (no visible trend in the split fraction). Noise shrinks with x (sd ~0.4 at x=0,
~0.1 at x=1).

## Model family
Mixture of two Normals, constant weight w:
- mA(x) = polynomial in x,
- mB(x) = mA(x) - max(x - x0, 0) * (d1 x + d0)   (hinge split, branches coincide left of x0),
- log sd = linear/quadratic in x, shared or per branch.
Fitted by maximum likelihood (L-BFGS + BFGS, random restarts).

## Validation
5-fold CV (one or two random fold splits), mean held-out NLL per row:
- A5, hinge-linear D, per-branch log-linear sd: 0.186 (2 reps)
- A5, constant D: 0.200 (worse -> D needs slope)
- A5, x-dependent weight (logit linear): 0.189 (no gain)
- A5, shared quadratic sd: 0.182 (2 reps)
- A4/A3/A5 with shared log-linear sd, A4 shared quadratic sd: 0.1795 / 0.1808 / 0.1807 / 0.1781 (1 rep)
Differences between these variants are ~0.003, well under the fold SE (~0.02).
Picked quartic mean, shared log-linear sd (13 parameters) as the simplest variant at the
top of the CV range. Train NLL 0.170; CV 0.180.

## Final
14 numeric constants. Expected held-out NLL is about the CV value, plus a little for
selection optimism.

expected_nll: 0.185
