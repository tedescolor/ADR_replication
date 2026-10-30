# Notes

## Data
1000 rows, x0 in [0,1]. Scatter: one noisy curve for x < ~0.57 (dips to -0.75 at x≈0.25,
rises to ~0.3 at 0.6). Above that it splits into two thin branches: the upper one rises to
~2.5 at x=1 and the lower one falls to ~-0.9. About 44% of points fall on the upper branch.
Noise sd drops steadily, from ~0.37 at x=0 to ~0.25 at x=0.5, and then to ~0.08-0.13 on
the branches near x=1. Residuals are close to Gaussian (kurtosis ~3).

## Validation
5-fold CV repeated with 2 shuffles (10 fits per config). Each config is fit by maximum
likelihood (BFGS, 4 random restarts). Score is the mean held-out NLL. Scripts are in work/.

## Candidates (CV NLL)
- Two free polynomial mixture components, cubic to quintic means and quadratic log-sd: 0.1975 to 0.209
- Shared mean poly m0 (deg K) + hinge r=max(x-c,0) with quadratic branch offsets, shared log-linear sd:
  K=3 0.1954, K=4 0.1963, K=5 0.1951; quadratic log-sd 0.1902; linear offsets 0.2035;
  cubic offsets 0.1970; Student-t 0.1976
- Same, plus a branch-specific sd slope in r (sep_s): K=4 0.1892, K=3 0.1898,
  K=4 with quadratic log-sd 0.1911
- K=4 sep_s with a **constant** mixture weight: **0.1892**. This is the final model.

## Final program
m0 = quartic(x); r = max(x-0.567, 0); upper = m0 + 0.335 r + 6.67 r^2;
lower = m0 - 5.99 r + 2.86 r^2; log sd = -0.98 - 0.77x - k_branch*r;
mixture weights 0.44/0.56. Before the split the two components are identical, so the model
is a single Gaussian there. 15 fitted constants (17 literals including the weights pair and 0.0).
Train NLL 0.1701, CV NLL 0.1892. I chose it because it had the lowest CV with the fewest parameters.

expected_nll: 0.19
