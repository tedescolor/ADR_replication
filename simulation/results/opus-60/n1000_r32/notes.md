# Notes

## Data
1000 rows, x0 in [0,1]. Scatter shows a single noisy curve for x < ~0.55 (dip to about -0.8 near
x=0.25). It then splits into two branches: an upper one rising almost linearly to about 2.5 at x=1, and a
lower one curving down to about -0.9. Noise sd falls from ~0.37 at x=0 to ~0.1 at x=1. Residuals are
close to Gaussian (excess kurtosis about -0.1).

## Model family
Two-component normal mixture:
- upper mean m1(x) = polynomial;
- lower mean m2 = m1 - (b1 r + b2 r^2), with r = max(x - c, 0). Both components coincide
  for x < c, giving a continuous bifurcation;
- log sd = polynomial in x (shared by both components or separate);
- constant mixture weight.
Fitted by direct MLE (BFGS).

## Validation
5-fold CV repeated over 2 seeds while screening, then 4 seeds (20 fits) for the finalists.
Mean held-out NLL:
- free polynomials for both means (deg 5/6), separate sds: 0.200–0.201
- split model, deg-5 mean, separate linear log-sds: 0.195; deg 4: 0.194
- linear split (b2=0): 0.211 (much worse); power split b*r^p: about the same as quadratic
- shared sd, deg-4 mean, linear log-sd: 0.193; quadratic log-sd: 0.192
- shared sd, **cubic mean**, quadratic log-sd, quadratic split: 0.1886 (2 seeds), 0.1905 (4 seeds)
- cubic mean, linear log-sd: 0.192; quadratic mean: 0.38 (underfits)
- an x-dependent weight or a cubic log-sd brought no gain.

## Final
Cubic m1, split at c=0.503 with a quadratic offset, shared quadratic log-sd, weight 0.44.
11 fitted constants (13 literals including the 0.0 / 1.0 helpers). Train NLL 0.1767,
CV NLL 0.1905. The model is small, so the gap is mostly fit variance; I use the CV value
plus a small margin.

expected_nll: 0.192
