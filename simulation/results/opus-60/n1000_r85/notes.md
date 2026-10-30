# Notes

## Data
1000 rows, x0 ~ U(0,1). For x < ~0.55, y follows one curved branch (dips to -0.8 near x=0.25),
with residual sd ~0.3 and slightly platykurtic noise. For larger x it splits into two branches of
roughly equal weight: an upper one rising almost linearly to 2.5 and a lower one peaking near 0.3
at x=0.65, then falling to -1. Noise sd shrinks toward x=1 (~0.1).

## Validation
5-fold CV repeated twice (two different permutations, 10 fits per config). Each fit is an MLE
from several BFGS starts. Score = mean held-out NLL on the original y scale.

## Candidates (CV NLL)
Free 2-component normal mixture, notation (mean deg, log-sd deg, logit-weight deg), u = 2x-1:
- (3,1,1) 0.243, (4,1,1) 0.234, (5,1,1) 0.234
- (3,2,2) 0.229, (4,1,2) 0.231, (4,2,0) 0.227, **(4,2,1) 0.2261**, (4,2,2) 0.2263, (4,3,2) 0.2256
- (5,2,2) 0.232, (6,2,2) 0.232, (5,3,3) 0.247 (overfitting / harder to optimize)
Structured "split" model (one quartic main curve, plus a lower branch = main minus a polynomial
in relu(x-c) with fitted c): 0.229-0.235. That is worse than the free mixture, which also uses
its second component on the left side, where it captures the flat-topped noise.

## Final
(4,2,1): both component means quartic in u, log-sds quadratic, mixture weight logit-linear.
27 constants. The train NLL is 0.2045 and the CV NLL is 0.226. I picked it over (4,3,2) because
it does just as well with fewer parameters. The final fit uses all 1000 rows, so I expect
roughly the CV value on new data.

expected_nll: 0.226
