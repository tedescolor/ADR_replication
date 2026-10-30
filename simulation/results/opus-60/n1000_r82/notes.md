# Notes

## Data
1000 rows, x0 in (0,1). Scatter shows two branches: for x < ~0.5 they overlap (wide, sd ~0.3-0.4),
for x > 0.5 they split: an upper branch rising almost linearly to ~2.5 at x=1 and a lower
branch peaking near 0.2 at x~0.65 and falling to -1 at x=1. Noise shrinks with x (sd ~0.1 at x=1).
Both branches have roughly half the points.

## Validation
5-fold CV (mean held-out NLL), repeated with two fold seeds; refit (BFGS, 3 starts) in each fold.

## Candidates (CV NLL)
- 2-comp Normal mixture, separate polys for means/log-sd/logit-weight: 0.243-0.245
- shared log-quadratic sd (both components): 0.236 — clearly better than separate sds (0.239-0.242)
- weight: constant = linear (0.2356) better than quadratic (0.237)
- mean degrees (upper, lower): (3,4) 0.2351/0.2342; (4,4) 0.2356/0.2362; (5,4) 0.2353/0.2346;
  (3,3) 0.237; log-sd degree 1 is worse (0.239), degree 3 no gain (0.238)
- Student-t components (nu fit ~22): 0.238, no gain
- sd linear (not log): same as log
- lower branch as a + b x + c sin(w x + phi) instead of quartic: 0.2353/0.2345, same as poly

Everything with reasonable flexibility gave ~0.235, so the remaining error is noise. I picked
the smallest model at that level.

## Final model
Mixture of 2 Normals with constant weights (0.559/0.441), upper-branch mean cubic in t=2x-1,
lower-branch mean quartic, shared sd = exp(quadratic in t). 13 fitted params (14 numbers incl. both weights), 16 constants in total incl. the 2.0, 1.0 rescaling.
Training NLL 0.2199, CV NLL ~0.235.

expected_nll: 0.237
