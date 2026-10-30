# Conditional density of y | x0

## Data
1000 rows, x0 in [0,1]. Scatter: for x0 < ~0.5 a single noisy curve (dip to about -0.8
at x0≈0.2). Past that it splits into two branches: an upper one rising to about 2.5 at
x0=1 and a lower one peaking near 0.1 at x0≈0.7 and falling to about -1. Noise shrinks
with x0 (sd ≈ 0.4 near 0, ≈ 0.11 near 1). Each branch is about 50% of the data.

## Model family
2-component Gaussian mixture, fitted by maximum likelihood (BFGS):
- component means: polynomials in x0 (both branches are defined everywhere; they overlap
  on the left, which reproduces the single wide curve there);
- scale: exp(a + b*x0), shared by both components;
- mixing weight: a constant, sigmoid(c).

## Validation
10-fold CV, repeated with 3 different shuffles (30 fits per config). Mean held-out NLL per row:

| means (upper,lower deg) | log-scale | weight | CV NLL |
|---|---|---|---|
| 3,3 | lin shared | const | 0.1992 |
| **3,4** | **lin shared** | **const** | **0.1910** |
| 3,5 | lin shared | const | 0.1915 |
| 4,4 | lin shared | const | 0.1922 |
| 5,5 | lin shared | const | 0.1909 (+3 params) |
| 4,3 | lin shared | const | 0.1968 |
| 3,4 | quad shared | const | 0.1919 |
| 3,4 | lin separate | const | 0.1913 |
| 3,4 | lin shared | logit-linear | 0.1917 |
| 3,4 | sd linear (not exp) | const | 0.1925 |
| 3,4 | constant | const | 0.2699 |
| poly + sin/cos(2πx) bases | lin shared | const | 0.199–0.226 |

On the right, where the branches are well separated, the standardized residuals look
Gaussian (excess kurtosis ≈ 0, skew ≈ -0.09), so heavier tails were not needed.

## Choice
Cubic upper mean, quartic lower mean, shared log-linear scale and a constant weight:
12 fitted constants (15 literals in the file, because the shared scale is written out twice and there is the 1 in 1 - w). This had the lowest CV of the small models,
and the larger models gave no meaningful gain. Refitted on all data: train NLL 0.1787.

## Estimate
The repeated 10-fold CV gives 0.191. I added a little for selection optimism.
expected_nll: 0.193
