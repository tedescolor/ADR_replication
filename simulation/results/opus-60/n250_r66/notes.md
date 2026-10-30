# Notes

## Data
250 rows, x0 in [0,1]. y follows a smooth curve (about -0.3 at x=0, minimum about -0.8 near x=0.22,
rising to 2.5 at x=1). Its noise shrinks as x grows (sd about 0.37 near 0, about 0.08 near 1).
For x > ~0.5 a second, lower branch appears: y near 0 at x=0.7, falling to about -0.9 at x=1.
It holds about 45% of the points there and is tight (sd 0.1-0.2).

## Model family
A two-component Gaussian mixture:
- branch 1: quartic polynomial mean; sd = |a + b x| (linear)
- branch 2: quadratic mean; sd = exp(c + d x)
- weight of branch 2: sigmoid(w0) * sigmoid(60 (x - x0)), a near-step switch-on at x0 ≈ 0.47
14 fitted constants (plus the fixed slope 60 and the 0.001 sd floor), all fitted by maximum likelihood.

## Validation
I used 5-fold CV repeated twice (10 folds). Each candidate was refit on every training fold,
with Nelder-Mead followed by BFGS. Mean held-out NLL (± standard error):
| variant | #par | train | CV |
|---|---|---|---|
| poly4 / exp-lin sd / logistic weight (base) | 13 | 0.185 | 0.245 ± 0.031 |
| + capped step weight (capw) | 14 | 0.159 | 0.226 |
| capw + quadratic log-sd | 15 | 0.155 | 0.235 |
| capw + quintic mean | 15 | 0.159 | 0.235 |
| capw + cubic branch 2 | 15 | 0.159 | 0.239 |
| capw + linear sd1 | 14 | 0.152 | 0.221 |
| capw + exp-lin sd2 | 15 | 0.144 | 0.206 |
| linear sd1 + exp-lin sd2 + free step slope | 15 | 0.139 | 0.199 |
| **same, step slope fixed at 60 (final)** | 14 | 0.142 | 0.199 ± 0.031 |

Adding more flexibility to the means made CV worse. The real gains came from modelling the branch
weight as a capped step and from letting both noise scales vary with x. I fixed the step slope
because the free fit sent it to infinity.

## Estimate
The CV NLL is 0.199. Choosing among about 10 variants adds a little selection optimism, and the
fold-to-fold standard error is about 0.03. I estimate:

expected_nll: 0.21
