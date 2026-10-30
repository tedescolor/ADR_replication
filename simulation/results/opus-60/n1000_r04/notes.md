# Notes

## Data
1000 rows, x0 in (0,1). Scatter shows a single noisy "trunk" curve (dips to ~-0.8 at x≈0.25,
rises through 0 at x≈0.5) which at x≈0.5 splits into two branches: the upper one keeps
rising to ~2.5 at x=1, the lower one bends down to ~-0.9. Noise sd shrinks with x
(≈0.4 at x=0 to ≈0.1 at x=1). Standardized residuals look Gaussian (|skew|,|kurt| small).

## Validation
5-fold CV (fixed random permutation), refitting by direct maximum likelihood
(multi-start Nelder-Mead + BFGS) in each fold. Per-point NLL sd ~1, so CV SE ~0.03;
differences < ~0.01 are not meaningful.

## Tried
- Unconstrained 2-normal mixture, polynomial means/log-sds/logit weight:
  (deg 3,1,1) CV 0.2035; (4,2,2) 0.2057; (5,2,2) 0.218; (5,3,3) 0.213. Left region is
  split artificially between components.
- Structured trunk + branch: upper mean = cubic trunk m(x); lower mean = m(x) + r(a + b r),
  r = max(x - x0, 0); shared sd exp(s0 + s1 x); constant weight.
  CV 0.2037 (10 params). Variants: deg-4 trunk 0.2043, cubic deviation 0.2067,
  separate branch sds 0.2042, quadratic log-sd 0.2029, x-dependent weight 0.2038.
  None better beyond noise.

## Choice
Structured trunk + branch model (10 fitted constants): it is the most interpretable,
matches the visible geometry (one curve before the split, two after), and is as good as
any richer variant in CV. Train NLL 0.1878, 5-fold CV NLL 0.2037.

expected_nll: 0.205
