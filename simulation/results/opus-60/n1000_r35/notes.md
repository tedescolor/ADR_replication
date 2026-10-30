# Notes

Data: x0 in [0,1], y. For x0 < ~0.46 a single noisy curve (dip to about -0.75 near x=0.22).
Above that it splits into two branches of roughly equal weight: one rises almost
linearly to 2.5 at x=1, the other stays flat and then falls to -1. Noise sd shrinks with x
(about 0.25 at x=0, about 0.1 at x=1). Within-branch residuals look Gaussian
(skew about 0, excess kurtosis 0.13).

## Final model
A 2-component Normal mixture:
- upper branch mean m1 = cubic in t = 2x-1
- lower branch mean m2 = m1 - max(x-c,0)^2 * quadratic(t). It equals m1 below the fitted
  split point c = 0.462, so the trunk is shared.
- a shared log-sd that is quadratic in t
- a constant mixture weight (0.534 / 0.466)

The program has 14 constants. It is fitted by direct MLE (L-BFGS-B, several inits and split-point starts).

## Validation
5-fold CV with a fixed permutation, refitting all parameters (including c) inside each fold.
Config is (mean deg+1, diff deg+1, logsd deg+1, weight deg+1):
- unconstrained mixture with two degree-5 polynomial means and log-linear sds: CV 0.1465
- split model (6,2,2,1): 0.1398; with separate branch sd: 0.1418; logistic weight: 0.1405
- (5,2,2,1) 0.1383, (7,2,2,1) 0.1403
- quadratic log-sd: (6,2,3,1) 0.1363, (6,3,3,1) 0.1332, (5,3,3,1) 0.1315, (6,3,4,1) 0.1317,
  (5,4,3,1) 0.1316, (5,3,3,2) 0.1320, (5,3,3,1)+separate sd 0.1323
- **(4,3,3,1) 0.1292** (chosen: it is the simplest and has the best CV), and (3,3,3,1) fails (0.36)

The in-sample NLL of the final model is 0.1162. Many configs land near 0.13, and the choice was
made on the same CV, so I add a small allowance for selection optimism.

expected_nll: 0.132
