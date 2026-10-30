# Notes

## Data
250 rows, x0 in (0,1). A single noisy curve for x < ~0.5 (dips to about -0.8 at x≈0.2),
which then splits into two branches: an upper branch rising to about 2.5 at x=1 and a lower
branch peaking near 0.2 at x≈0.7 and falling to about -1 at x=1. Roughly half of the points
for x > 0.5 sit on each branch. Noise sd falls from about 0.37 at x=0 to about 0.12 at x=1.

## Validation
5-fold CV repeated 3 times (different shuffles), refitting by maximum likelihood
(BFGS + Powell, several starts) on each training split, then averaging held-out NLL over
all 750 held-out predictions. SE of the mean is about 0.05 (dominated by between-point variance;
differences between models are paired, so they are more precise than that).

## Candidates (train NLL / CV NLL)
- Two Gaussians, logistic gate, cubic main mean, linear lower branch: 0.246 / 0.343
- Same with quadratic lower branch (A, 12 params): 0.160 / 0.237–0.242; cubic lower branch: 0.239; Student-t: 0.247
- Capped gate (B): 0.137 / 0.209; plus shared sd function (C): 0.134 / 0.193; separate linear sd (D) 0.211
- **Lower branch = main curve - a*r - b*r^2, r = max(x-c,0)**, shared sd, gated (F): 0.134 / 0.180
  The fitted gate was effectively constant, so:
- **H: same with constant weight (10 params): 0.134 / 0.176**  <- chosen
- Variations of H: tangent branching only (b only) 0.181; extra cubic deviation 0.180;
  separate sd scale for the lower branch 0.185; quartic main curve 0.185; quadratic log-sd 0.183.

## Final program
mixture of 2 Normals with shared scale s = exp(a0 + a1 x):
  m1 = cubic in x; m2 = m1 - r(b1 + b2 r), r = max(x - 0.511, 0); weight of lower branch 0.459.
(Left of the branch point both components coincide, so it is one Gaussian there.)
10 fitted constants (11 literals including the 0.0 in the max). Full-data train NLL 0.134.
Residuals look Gaussian (skew -0.25, excess kurtosis 0.17) with no remaining trend.

Expected NLL on new data: the CV estimate 0.176 (the full-data fit uses 25% more data, so it may be slightly
better); I report about 0.175.

expected_nll: 0.175
