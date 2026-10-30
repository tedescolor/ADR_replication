# Notes

## Data
1000 rows, x0 in (0,1), y in (-1.6, 2.6). Scatter plot: for x0 < ~0.5 there is a single
noisy wavy curve (sd ~0.33 at x0=0, falling to ~0.2 at 0.5). From about x0=0.5 the data
**splits into two branches**: one rises almost linearly to y≈2.5 at x0=1, and one bends down to
y≈-1 at x0=1. The branches are roughly equally populated, and their noise shrinks to ~0.1–0.15.
Within the bins the residuals are close to Gaussian (|kurtosis| is small once the bimodality is removed).

## Validation
5-fold CV repeated twice (10 folds, fixed seeds), refitting by maximum likelihood (L-BFGS-B)
on each training split. Each fold is fitted from two starts (a data-driven init and the full-data solution) and the better
training objective is kept. The figure reported is the mean held-out NLL per row, ± SE over folds.

## Candidates (CV NLL)
- Unstructured 2-Gaussian mixture with polynomial means/log-sds/logit-weight in x:
  0.217–0.226 (22–29 params). The components disagree in the single-branch region, which wastes parameters.
- Structured "shared trend + branch split" (the final family): base(x) polynomial in t=2x-1,
  h=max(x-c,0), upper/lower means = base + poly(h) (no intercept), log-sd = linear(t) + k·h
  per branch, mixture weight fixed at 0.5 (a free weight or a weight that depends on h did not help).
  Grid over c ∈ {0.45,0.5,0.51,0.53,0.55,0.57,0.6} and over the degrees:
  - c=0.5, cubic base, quadratic branches, log-sd linear: **0.2121 ± 0.013** (13 constants)
  - c=0.51: 0.2124; c=0.53: 0.2144; c=0.55: 0.218–0.222; c=0.6: 0.236
  - Higher degrees (quartic base, cubic branches, quadratic log-sd) gave no improvement.
  - Lower degrees (quadratic base, constant log-sd, no shrinking of the branch sds) gave 0.217–0.30.
- A smooth (softplus) change-point with a free location and BFGS was numerically unstable,
  so I dropped it in favour of the grid over c.

## Final program
Equal-weight mixture of two Normals. Shared cubic trend in t, hinge at x0=0.5, and quadratic
offsets in h for each branch. Log-sd = -0.3265 t - 1.355 - k·h, with k different for each branch. 13 numeric
constants. Train NLL 0.2002, CV NLL 0.2121. The PIT/KS checks in 10 x-bins show no
significant miscalibration.

## Expected NLL
The CV estimate is 0.212 (SE ≈ 0.013). I add a little for the selection optimism from the ~20
configurations compared.

expected_nll: 0.215
