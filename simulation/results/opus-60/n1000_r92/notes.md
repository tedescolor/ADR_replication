# Notes

## Data
1000 rows, x0 in [0,1]. y is unimodal-ish noisy curve (sd ~0.3) for x<0.45, then splits
into two branches: one rising roughly linearly to ~2.5 at x=1, one peaking ~0.2 at x~0.7 and
falling to ~-1 at x=1. Noise shrinks with x (~0.3 -> ~0.08). Fewer points on the upper branch at high x.

## Model family
2-component Gaussian mixture, all in t = 2*x0 - 1:
- component means: polynomials of degree dm,
- log-sd: polynomial of degree ds (separate per component, or shared),
- mixture weight: sigmoid(polynomial of degree dw).
Fitted by direct MLE (L-BFGS-B, 3 restarts from a branch-shaped initialisation).

## Validation
5-fold CV (fixed random split), mean held-out NLL:
| config (dm,ds,dw) | separate sd | shared sd |
|---|---|---|
| 4,1,1 | 0.1881 | |
| 5,1,0 | 0.1908 | |
| 5,0,1 | 0.2403 | |
| 5,1,1 | 0.1865 | 0.1859 |
| 5,2,1 | 0.1878 | 0.1815 |
| 6,1,1 | 0.1852 | |
| 6,2,1 | 0.1867 | **0.1793** |
| 5,3,1 | | 0.1808 |
| 6,3,1 | | 0.1799 |
| 7,2,1 | | 0.1830 |
| 7,2,2 / 8,2,2 / 7,3,2 / 9,3,3 | 0.19-0.23 | |
Shared log-sd with a quadratic is consistently better than separate sds; higher mean
degrees / weight degrees overfit or hit local optima.

## Final
Shared-sd (6,2,1): 2 normal components, degree-6 means, quadratic log-sd, linear-logit weight.
23 constants. Train NLL 0.1557, CV NLL 0.1793. Refit on all data adds ~25% more training
data, so held-out NLL should be at or slightly below CV; I add a small margin for selection
among ~15 configs.

expected_nll: 0.18
