# Conditional density estimation — notes

## Data
1000 rows, x0 in [0,1]. Scatter shows two branches of y that cross around x≈0.5:
one dips to about -0.8 near x≈0.25 then rises roughly linearly to 2.5 at x=1; the other
rises slowly to about 0.2 near x≈0.65 then falls to -1 at x=1. At small x the branches overlap.
Noise looks Gaussian within each branch (kurtosis of binned residuals ≈ 0). Its sd is ≈0.4 at
x=0, ≈0.25 in the middle and ≈0.08 at x=1.

## Validation
All structures were fitted by maximum likelihood (BFGS with restarts) and compared by
K-fold CV: 5-fold × 2 repeats, then 10-fold × 3 for the final model. Folds were paired across
models. The fold-level SE is ≈0.013, but paired differences are much smaller.

| model | params | train NLL | CV NLL |
|---|---|---|---|
| 2 normal comps, poly means (5,3), log-linear sd, logistic weight | 16 | 0.188 | 0.208 |
| shared-then-forking mean (softplus fork), sep. sd, sigmoid weight | 17 | 0.181 | 0.199 |
| same, shared log-cubic sd | 15 | 0.181 | 0.198 |
| two free poly-5 curves, free sds/const weight | 19 | 0.179 | 0.202 |
| Student-t components | 21 | 0.182 | 0.203 |
| **two free poly-5 curves in z=2x-1, weight fixed 0.5, shared log-cubic sd** | **16** | **0.176** | **0.195** (10×3 CV: 0.1947) |
| same with degrees 6/6, 7/7, 6/5, 5/6 | 17-20 | ~0.176 | 0.196-0.200 |
| shared sd log-quadratic / log-quartic | 15/17 | 0.180/0.176 | 0.197/0.195 |
| sinusoid+linear means | 14 | 0.189 | 0.202 |

When the mixture weight was free it was estimated at 0.51, so it is fixed at 0.5. Using the
centred variable z=2x-1 made the optimisation much better conditioned.

## Final program
Mixture of 0.5·N(mA(z), s(z)) + 0.5·N(mB(z), s(z)), with z = 2x-1, mA and mB degree-5
polynomials, and log s a cubic. 19 literals in total: 16 fitted constants plus 2.0, 1.0 and the
0.5 weights. I picked it because it has the lowest CV of the candidates and is among the
simplest; adding degrees or separate sds does not help out of sample.

Expected NLL on new data: about the CV value, 0.195. The full-data refit should do slightly
better than the fold fits, but the test set adds sampling noise of ±0.013.

expected_nll: 0.195
