# Notes

## Data
250 rows, x0 in (0,1). For x < ~0.5 y is a wide (sd ~0.33), platykurtic band dipping to about -0.9
near x=0.25. Above x≈0.55 it clearly splits into two thin branches (sd ~0.1): an upper one rising
almost linearly to about 2.5 at x=1, and a lower one that stays flat near 0.1 and then falls to -0.9.
The negative kurtosis at low x suggests two overlapping curves everywhere, not one trunk.

## Tried (5-fold CV, 2 repeats, mean held-out NLL)
- Free 2-component Gaussian mixtures with polynomial means, log-sd and logit weight, random inits:
  unstable, CV 0.21–0.25.
- Same family with data-driven inits (from trunk+upper and trunk+lower polyfits). Varying the
  degrees (means 3–5, sd 0–2, weight 0–2, shared vs separate sd) gave CV 0.206–0.30; constant sd was
  clearly worse (0.28–0.30), and higher-degree means overfit.
- A "fork" model (shared trunk + hinge branches after a fitted split point): CV 0.26–0.47, worse.
  The hinge made optimisation harder, and the shared-trunk assumption seems wrong.

## Final
Mixture of 2 Normals: cubic mean per component, shared sd = exp(quadratic in x), and
weight = sigmoid(quadratic in x). 14 fitted constants (plus the 1.0 in `1 - w`). Fitted by maximum likelihood (BFGS) on all data.
Train NLL 0.143; 5-fold x2 CV 0.206; 10-fold x3 CV 0.197 (per-point SE ~0.046).
I chose it because it had the best CV among the small models. The differences between the leading
specs are within noise, so I kept the simple one.

expected_nll: 0.21
