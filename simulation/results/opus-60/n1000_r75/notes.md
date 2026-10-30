# Notes

## Data
1000 rows, x0 roughly uniform on [0,1]. y follows a single noisy U-shaped trunk for small x,
then splits into two branches of about equal mass: the upper one rises almost linearly to ~2.5
at x=1, the lower one bends down to ~-1. Residuals are Gaussian (skew ~0, excess kurtosis ~0),
and the noise sd shrinks with x: ~0.37 at x=0, ~0.22 at 0.45, ~0.1 at x=1, about the same on
both branches.

## Validation
5-fold CV with a fixed permutation, repeated with 2 seeds for the final candidates.
Each training fold was refit from several starting points. Scores are mean held-out NLL.

## Candidates (CV NLL)
- Free 2-component Gaussian mixture: polynomial means (deg m), polynomial log-sd (deg s),
  polynomial logit weight (deg w):
  (3,1,1) 0.210, (4,1,1) 0.211, (4,2,1) 0.207 (17 params).
- Structured "trunk + split" mixture: mu1 = poly(x); mu2 = mu1 + g*(b0 + b1*g) with
  g = softplus-smoothed (x-c)_+; shared log-sd poly; one weight. The degrees below are
  (trunk mean, log-sd); "sep sd" gives each branch its own log-sd:
  (5,1) 0.212, (5,2) 0.208, (4,2) 0.207, (6,2) 0.207, (5,3) 0.210.
  For the best fits the softplus collapsed to a hard kink and the weight to 0.5.
  The sep-sd variant had been unstable earlier (two of three runs hit bad optima) and was
  not pursued.
- Final: the same model with a hard kink g = max(x-c,0), shared sd, and equal weights,
  refit with Powell + Nelder-Mead from 8 starts: **CV NLL 0.2036** over 2x5 folds
  (fold values 0.14-0.26, so the standard error is about 0.013). Train NLL 0.1945.

## Final program
Mixture of two equally weighted Normals with a shared sd:
- branch 1 mean: quartic in x (5 constants);
- branch 2 mean: branch 1 + g*(b0 + b1*g), g = max(x - c, 0) (3 constants);
- sd: exp(quadratic in x) (3 constants).
That is 11 fitted constants (12 literals counting the 0.0 in the max and the two 1.0
weights: 14 literals). The fitted c is about 0.2, below the visible split near 0.55. That is
fine because b0 < 0 and b1 > 0 make the gap g*(b0+b1*g) close to zero until about x = 0.47,
after which it grows quickly.
I chose it because it ties or beats every richer variant on CV while having the fewest
constants, and it is a direct description of the structure in the data.

expected_nll: 0.205
