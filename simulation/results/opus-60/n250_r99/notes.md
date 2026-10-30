# Notes

## Data
250 rows, x0 in (0,1). Scatter shows one noisy branch (sd ~0.3) for x < ~0.55: it dips to
about -0.8 at x~0.22, then rises to ~2.5 at x=1. Above x~0.6 a second, tighter lower branch splits
off (~0 at 0.65-0.75, falling to ~-0.9 at x=1), carrying about half the mass.

## Models tried (5-fold CV x 2 repeats, refit with multi-start BFGS on each training fold)
- Single heteroscedastic Normal (poly-5 mean, poly-2 log-sd): CV NLL 0.63.
- Generic 2-component Gaussian mixture with polynomial means/log-sds and logistic weights,
  random init (2,3,1,2): CV 0.22; fitted components did not match the visible branches.
- Structured "gated branch" mixture: main Normal (poly mean, poly log-sd) everywhere, plus a
  lower-branch Normal whose weight is a*sigmoid(k(x-c)). (mean deg, log-sd deg, branch mean deg,
  branch log-sd deg):
  - (4,1,2,0) 0.231, (5,1,2,0) 0.231, (5,2,2,0) 0.218, (6,2,2,0) 0.216
  - (5,2,2,1) **0.190** (se ~0.03), (5,1,2,1) 0.198, (6,2,2,1) 0.195, (5,2,2,2) 0.219,
    (5,1,3,1) 0.197, (7,2,2,0) 0.55 (unstable)
  - adding a nonzero floor to the gate weight (lower branch also present at small x): 0.194, no gain.

## Final
Gated mixture (5,2,2,1): main branch quintic mean, quadratic log-sd; lower branch quadratic
mean, linear log-sd; weight sigmoid(a)*sigmoid(k(x-c)). 17 constants. Train NLL 0.119, CV NLL 0.190.
Chosen as the best CV among ~12 candidates; neighbours give 0.195-0.198, so the result
holds up under small changes. I added a small allowance for selection optimism.

expected_nll: 0.20
