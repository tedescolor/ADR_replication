# Notes

## Data
1000 rows, x0 in [0,1], y in [-1.65, 2.59]. The scatter plot shows two crossing curves.
One branch dips to about -1 near x=0.2 and then rises to 2.5 at x=1. The other starts near
0.4, falls to -0.7, rises a little to about 0.1 near x=0.7 and drops to -1 at x=1. The
noise shrinks as x grows (sd about 0.28 at small x, 0.09 at x=1). Within a branch the
noise looks Gaussian.

## Validation
Each model was fitted by maximum likelihood (BFGS on the mean NLL) and scored by 5-fold
CV: 3 repeats with different shuffles, so 15 held-out folds. The fold standard error is
about 0.03, so differences under ~0.01 are noise and I preferred the simpler model.

## What I tried (CV mean NLL)
- 2-component Gaussian mixture, polynomial means, separate log-polynomial scales and a
  logistic-polynomial weight: 0.211-0.221 (degrees 3-6).
- Shared scale for both components: 0.208.
- Constant weight (0.208) vs. fixed equal weights 0.5/0.5 (0.207). The fitted weights
  were close to 0.5 anyway.
- Mean degree, with equal weights and shared log-quadratic scale:
  quartic/quartic 0.2034 (best), 3/4 0.206, 4/5 0.2055, 5/5 0.207, 6/6 0.208,
  3/3 0.216.
- Scale degree: linear 0.210, quadratic 0.203, cubic 0.209.
- Student-t components: the fitted df went to about 2400, so Gaussian is right.

## Final program
Mixture of two equal-weight Normals. Each mean is a quartic in x. Both share the scale
exp(quadratic in x). There are 13 fitted parameters and 22 literals once you count the
exponents and the two 0.5 weights. It was refitted on all the data: training NLL 0.187,
CV 0.203.

expected_nll: 0.205
