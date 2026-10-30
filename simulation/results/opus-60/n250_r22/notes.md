# Notes

## Data
250 rows, x0 in [0, 1]. The scatter plot shows two branches. For x > 0.55 they separate
clearly: an upper branch that rises almost linearly to about 2.6, and a lower branch that
peaks near 0.1 at x ≈ 0.7 and falls to -1. For x < 0.45 the two branches overlap around -0.6,
and the spread there is wider (sd ≈ 0.35). The noise around each branch shrinks as x grows
(≈ 0.13 at x = 1).

## Validation
10-fold CV repeated over 2 random splits: refit by MLE (BFGS, several restarts) on each
training fold, then score the mean NLL on the held-out fold. All model choices were made
on CV NLL. Final constants come from a fit on all 250 rows.

## Candidates (CV NLL, # params)
- 2-Gaussian mixture, cubic upper / quadratic lower mean, log-linear sds, logistic weight: 0.328 (13)
- both means cubic, separate log-linear sds, logistic weight: 0.310 (14); constant weight: 0.304 (13)
- quartic/quintic means, quadratic log-sd, x-dependent weights: 0.303–0.340 (no gain, some overfitting)
- Student-t components: df → ∞, no gain (0.304)
- shared sd for both components: log-linear 0.296 (11); log-quadratic 0.290 (12)
- shared log sd = a + b·x²: 0.2864 (11)
- the same with one shared intercept for both mean cubics (the branches meet at x = 0): **0.2847 (10)**
- variants (sd offset between components, cubic log-sd, linear-logit weight, quartic mean): 0.291–0.297

## Final program
A mixture of two Gaussians with fixed weights 0.595 / 0.405. Both means are cubics in x with
a shared intercept. Both components use the same sd, exp(a + b·x²). That is 10 fitted constants
(11 literals including the second weight). Training NLL is 0.243 and CV NLL is 0.285. The final
fit uses all the data, which should help a little, but some selection optimism remains, so my
estimate is about the CV value.

expected_nll: 0.29
