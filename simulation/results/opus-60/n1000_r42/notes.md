# Notes

## Data
1000 rows, x0 ~ U(0,1), y continuous. Scatter: for x < ~0.54 a single noisy curve
(dips to about -0.9 at x≈0.25). Above that point it splits into two branches: an upper one
rising to about 2.5 at x=1 and a lower one bending down to about -1. The noise sd falls from
~0.36 (flat for x<0.3) to ~0.09 at x=1. Residuals in the single-branch region look
Gaussian (skew ≈ 0, excess kurtosis ≈ 0.3).

## Validation
5-fold CV repeated twice (10 folds, fixed seeds), with the model refit by BFGS MLE on each
training fold. I report the mean held-out NLL with its standard error (≈0.015, mostly
fold-to-fold noise).

## Candidates (CV NLL)
- 2-comp Normal mixture, free quartic means, log-linear sds, logistic weight (16 params): 0.206
- Structured: lower mean = upper mean − gap(h), with h = softplus hinge at x0 (scale fitted)
  and gap = h(g1+g2 h); shared sd; constant weight
  - log-linear sd: 0.199; separate sds: 0.203; x-dependent weight: 0.1997
  - **log-quadratic sd (13 params): 0.1894**
  - StudentT noise: 0.1896 (ν large → Normal is enough)
  - sinusoidal upper mean: 0.1894; sin+quadratic: 0.1899; poly deg 5: 0.1901
  - log-cubic sd: 0.1894; hinge-shaped sd: 0.1886; linear/quadratic sd (not log): 0.191–0.192

## Final
Upper-branch mean is a quartic (5 coefs). Lower branch = upper − h(5.44 + 4.77h), where h is a
softplus hinge at x=0.544 with width 0.019. Both branches share a Normal sd of
exp(quadratic in x). Branch weight is a constant 0.577. That is 13 fitted parameters,
plus a few structural literals. I chose it because it is the simplest of the
models tied at the best CV; the ones that scored slightly better were within one standard
error. Train NLL is 0.174 and CV NLL is 0.189.

expected_nll: 0.19
