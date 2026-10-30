# Notes

## Data
250 rows, x0 in [0,1]. For small x0, y follows one noisy curve: about -0.2 at 0, dipping to
-0.7 near 0.2 and rising to ~0.15 near 0.65. The noise sd shrinks with x, from ~0.45 to ~0.1.
At high x0, y clearly splits into two branches. The upper branch rises from ~1.1 at 0.64 to
~2.5 at 1 with sd ~0.13. The lower branch falls to about -1 at x0=1. About half the points
are in each branch. Between 0.35 and 0.6 the two branches overlap, and the maximum-likelihood
fit puts the start of the upper branch at x0 ~ 0.34. There, the upper branch is a quadratic
that starts near -0.9.

## Model family
A two-component Normal mixture:
- lower branch: mean is a polynomial of degree dA in x; log sd is a polynomial of degree dS;
- upper branch: mean is a polynomial of degree dB; log sd is a polynomial of degree dSB;
- upper-branch weight = sigmoid(wm) * sigmoid(k (x - g0)), with g0 and k fitted (k comes out
  very large, so it is effectively a step).
I also tried a Student-t lower component.

## Validation
- 10-fold CV repeated twice (500 held-out predictions), with mean held-out NLL.
- Each training fold was fitted by BFGS from 3 initial gate positions (0.35, 0.5, 0.62), and
  the run with the best training likelihood was kept. The likelihood has several local optima,
  so the choice of starting points matters.
- Standard error of the CV mean is ~0.05. Most of that comes from point-to-point
  variability, so paired differences between models are smaller than that.

| dA,dS,dB,dSB | train | CV |
|---|---|---|
| 4,2,2,0 | 0.230 | 0.303 |
| 3,2,2,0 | 0.236 | 0.309 |
| **4,1,2,0** | 0.231 | **0.296** |
| 4,2,1,0 | 0.251 | 0.345 |
| 5,2,2,0 | 0.233 | 0.314 |
| 4,2,2,1 | 0.226 | 0.299 |
| 4,1,2,1 | 0.227 | 0.313 |
| 4,1,3,0 | 0.245 | 0.337 |
| 4,0,2,0 | 0.315 | 0.374 |
| 4,1,2,0 with t | 0.231 | 0.295 |
| 3,1,2,1 | 0.232 | 0.294 |
| 5,1,2,1 | 0.227 | 0.319 |

The top models are within noise of each other. I chose (4,1,2,0) Normal: it has 13 fitted
parameters and among the best CV scores. The t variant was no better once its extra
parameter is counted. In model.py the gate steepness is capped at 1e4 (the fit gave ~3e6),
which raised the training NLL from 0.2306 to 0.2319.

## Expected performance
CV is 0.296 ± 0.05. I add a small allowance for the high-variance region at low x and for
the uncertain branch onset near x0 ~ 0.34–0.6.

expected_nll: 0.31
