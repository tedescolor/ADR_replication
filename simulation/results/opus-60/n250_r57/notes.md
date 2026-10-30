# Notes

## Data
250 rows, x0 in [0,1]. Scatter shows two branches: a curved, wavy lower branch over the
whole range (min about -0.8 near x=0.25, about 0 at x=0.8, about -0.9 at x=1), and from
x~0.5 on, a tight linear upper branch (y ~ -2.67 + 5.36 x, sd ~0.11). The two branches overlap near x~0.5.

## Model family
A 2-component Normal mixture. The upper-branch weight is sigmoid(a + b x).
Lower branch: polynomial or sine mean, log-polynomial scale. Upper branch: linear mean.
Fitted by maximum likelihood (BFGS + Nelder-Mead, multiple random restarts).

## Validation
5-fold CV repeated twice (10 folds; 8 restarts per fit). Mean held-out NLL (SE):
- A3: cubic lower mean, quadratic log-sd lower, linear log-sd upper: 0.327 (0.031)
- A4: same with quartic mean: 0.302 (0.035), train 0.219
- A5: quintic: 0.311 (0.035)
- B: sine + linear lower mean: 0.315 (0.033)
- S: A4 with SkewNormal lower: 0.310 (0.039)
- T: A4 with Student-t lower: df -> infinity (collapses to Normal), no gain
- **C: quartic lower mean, log-linear lower sd, constant upper sd: 0.278 (0.029)**, train 0.231

## Choice
C has the best CV and the fewest parameters (12 fitted constants + the literal 1.0 = 13 literals).
Final constants are refit on all data.

## Expected NLL
CV gives 0.278. I add a little for the selection among ~7 variants and for the noisy SE.
expected_nll: 0.29
