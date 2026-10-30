# Notes

## Data
1000 rows, x0 in [0,1]. Scatter shows one noisy curve for x < ~0.5 that splits into two
branches for x > ~0.5: an upper one rising to ~2.5 at x=1 and a lower one bending down to
~-1. Noise sd decreases with x (~0.37 at x=0 to ~0.09 at x=1). Residuals within branches
look Gaussian (skew -0.09, excess kurtosis 0.06). Branch split is ~53/47 and constant in x.

## Validation
5-fold CV (fixed permutation), paired per-row comparisons of held-out NLL between
candidates (reporting mean difference +- SE). Fits by BFGS on full likelihood, several
random restarts.

## Candidates (CV NLL)
- 2 free Gaussian components, quintic means, separate log-linear sds, logistic weight: 0.216
- same, shared log-quadratic sd, constant weight: 0.210
- "split" model: lower mean = polynomial, upper = lower + a r^2 + b r with
  r = softplus hinge at x~0.49; shared log-quadratic sd; constant weight:
  deg 5: 0.210, deg 7: 0.208, deg 8: 0.206, deg 9: 0.2016, deg 11: 0.2024
  (using centered u = 2x-1 fixed ill-conditioning of the monomial basis)
- sinusoidal lower branch (fewer params): 0.214-0.218
- separate sds per branch, x-dependent weight, linear sd: no gain.

## Final
Split model, degree-9 lower polynomial in u=2x-1, softplus hinge (sharpness 15) for the
branch separation, shared sd = exp(quadratic in u), constant weight 0.531.
17 fitted parameters, 23 literal constants. Train NLL 0.1836, CV NLL 0.2016 (SE of
per-row mean ~0.024; paired advantage over deg 7 = 0.0076 +- 0.004).
Expected NLL on new data, adding a little for model-selection optimism:

expected_nll: 0.205
