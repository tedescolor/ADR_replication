# Notes

## Data
1000 rows, x0 roughly uniform on [0,1]. For x0 < ~0.5, y follows one noisy curve (sd ~0.35, shrinking with x).
After that it splits into two branches: an upper one that rises roughly linearly to ~2.5 at x=1, and a lower one
that peaks near 0.2 at x≈0.7 and falls to ~-1. Noise shrinks to sd ~0.1 at x=1. Residuals within a branch
look Gaussian (skew ≈ -0.1, excess kurtosis 0.15–0.45).

## Validation
5-fold CV repeated over 2 random partitions (10 fits). Each fit is a BFGS MLE with several random restarts.
The score is the mean held-out NLL.

## Candidates (CV NLL)
- 2-comp normal mixture, independent polynomial means (deg 3/4/5/6), log-linear or quadratic sd per component,
  constant or logistic weight: 0.185–0.199 (deg 4, shared sd: 0.1824)
- shared cubic base + softplus hinge with quadratic per-branch deviation: 0.1831–0.1883
- **shared cubic base f(x) + ReLU hinge at t: upper = f - a h², lower = f + b h + c h², h = max(x-t,0);
  shared log-linear sd; constant weight (11 params): 0.1805**
- variants of the above: sd linear in x 0.1804, quadratic log-sd 0.1808, logistic weight 0.1812, separate sd 0.1833,
  extra cubic hinge term 0.1828.

Those variants are all within CV noise of each other. I kept the simplest one (11 constants, sd = exp(linear),
which stays positive for any x). Its training NLL on all 1000 rows is 0.1696.

## Final
`model.py`: 11 fitted constants (plus the structural constants 2, 1 and 0). It passes the contract check.
Expected held-out NLL ≈ CV value (0.1805). The final model is fitted on 25% more data, so I report a slightly lower figure.

expected_nll: 0.180
