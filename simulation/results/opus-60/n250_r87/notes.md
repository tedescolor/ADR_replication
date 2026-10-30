# Notes

## Data
250 rows, x0 in (0,1). For x0 < ~0.55, y has one noisy branch (sd ~0.3) that dips to about -0.75
near x0 = 0.25 and comes back to 0 near x0 = 0.5. Above ~0.55, y splits into two clear branches:
an upper one that rises to about 2.5 at x0 = 1 with small noise (sd ~0.08), and a lower one that
falls to about -0.8.

## Model family
A 2-component Normal mixture in t = 2*x0 - 1:
- each component mean is a polynomial of degree dm in t,
- each component log-sd is a polynomial of degree ds,
- the mixing weight is sigmoid(polynomial of degree dw).
Fit by maximum likelihood (L-BFGS-B then BFGS, several random restarts).

## Validation
5-fold CV, repeated with 2 different random splits (mean held-out NLL per row):

| (dm, ds, dw) | train | CV |
|---|---|---|
| (3,1,1) | 0.167 | 0.250 |
| (4,1,1) | 0.139 | 0.206 |
| (5,1,1) | 0.136 | 0.221 |
| (6,1,1) | 0.132 | 0.217 |
| (5,1,0) | 0.145 | 0.210 |
| (5,2,1) | 0.102 | 0.296 |
| (5,2,2) | 0.079 | 0.273 |

Quadratic log-sd or quadratic weight overfits badly. (4,1,1) has the best CV, so I chose it.
In the fitted model the upper component has weight about 0.08 at x0 = 0 and about 0.65 at
x0 = 1, so the low-x region is mostly unimodal, which matches what the plots show.

## Final
(4,1,1), refit on all data: 16 constants (plus the 2 in t = 2*x - 1, so 18 literals). Train NLL is 0.139.
CV over different splits gives 0.20 to 0.21, so I add a small margin for the extra noise.

expected_nll: 0.22
