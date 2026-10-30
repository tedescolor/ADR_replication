# Notes

## Data
1000 rows, x0 in [0,1]. For low x, y follows one noisy curve (sd about 0.37 at x=0, shrinking with x).
Around x≈0.47 (u = 2x-1 ≈ -0.06) the curve splits into two branches of roughly equal
weight: one rises to about 2.5 at x=1, the other bends down to about -1. Within each branch
the residuals are close to Gaussian (pooled skew 0.1, excess kurtosis about 0), and both
branches have roughly the same sd.

## Validation
5-fold CV (fixed permutation, seed 0), with the full maximum-likelihood fit
(BFGS, several starts) redone inside each fold. I report the mean held-out NLL and its
standard error (about 0.023 across points; differences between models are paired, so
they are resolved better than that).

## Candidates (CV NLL)
- 2-Normal mixture with a sigmoid weight (capped at 0.5), polynomial means, and per-component
  log-sd polynomials, fitted on raw x: degrees 5/1/1 0.253, 5/2/1 0.237, 6/2/1 0.230, 6/2/2 0.227.
  At higher degrees the optimiser had trouble on raw x.
- Shared sd, centered u=2x-1: 6/2/2 0.226, 6/3/2 0.2238, 7/3/2 0.2257, 6/4/2 0.2261.
  Problem: the fitted weight was a near-step at x≈0.29 (or at x≈0.07 when the slope was
  bounded), which puts a visible artifact into the density. The extra component was being
  used as extra shape for the main curve rather than to model the branch.
- **Final**: weight fixed at 0.5. m2 = m1 + (d2*g + d1)*g with g = softplus(k(u-c))/k, so
  the components coincide before the split and the density is smooth.
  Degrees (mean/gap/log-sd) 6/1/2 0.2280, **6/2/2 0.2236**, 6/3/2 0.2248, 7/2/2 0.2237,
  6/2/3 0.2253, 5/2/2 0.2260.

I chose 6/2/2: it has the best CV (tied), 15 fitted parameters (about 20 literals), a smooth
density with no artifacts, and it is interpretable. Training NLL on all data is 0.2076.

## Estimate
CV NLL 0.2236 (se about 0.023). The final model is fitted on all 1000 rows, so it is slightly
better than the 800-row fold fits. Allowing for some selection optimism, I estimate:

expected_nll: 0.225
