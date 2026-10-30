# Notes

## Data
250 rows, x0 in (0,1). Scatter shows a single noisy band for x < ~0.6 (dip to about -0.8 near
x=0.25, sd about 0.3-0.5, larger near x=0) that splits into two tight branches (sd about 0.1) for x > ~0.65:
an upper branch rising almost linearly to about 2.5 at x=1, and a lower branch that stays near 0.1 and then
bends down to about -1.15 at x=1.

## Model family
A mixture-of-experts of Gaussians: K components, each with a polynomial mean in t = x-0.5 (degree dm),
a log-sd polynomial (degree ds), and softmax weights with polynomial logits (degree dw). The models were
fitted by penalized maximum likelihood (L-BFGS, small L2 penalty on the coefficients, random restarts).

## Validation
5-fold CV (one fixed permutation, 6 restarts per fold), mean held-out NLL:

| K,dm,ds,dw | params | l2 | CV NLL |
|---|---|---|---|
| 2,3,2,2 | 17 | 1e-3 | **0.287** |
| 2,4,3,3 | 22 | 1e-3 | 0.301 |
| 3,3,1,1 | 22 | 1e-3 | 0.311 |
| 2,5,2,2 | 21 | 1e-3 | 0.313 |
| 2,4,2,2 | 19 | 1e-3 | 0.315 |
| 2,3,1,1 | 14 | 1e-3 | 0.320 |
| 2,3,2,2 | 17 | 0.1 | 0.306 |
| 2,4,2,2 | 19 | 0.1 | 0.308 |
| 3,3,2,2 | 30 | 0.1 | 0.347 |
| 2,3,2,2 | 17 | 1 | 0.549 |

## Final
I chose (2,3,2,2) with l2=1e-3: 2 Gaussian components, cubic means, quadratic log-sds and a quadratic
weight logit, for 17 constants (plus 0.5 for centring x and the 0.0 reference logit). Refitting on all
the data gives a training NLL of 0.196, and a second refit with 12 restarts reached the same optimum.
The CV estimate (0.287) comes from models trained on 80% of the data, which makes it slightly
pessimistic. Fold-to-fold optimizer variance makes it noisy in both directions, so I report about 0.29.

expected_nll: 0.29
