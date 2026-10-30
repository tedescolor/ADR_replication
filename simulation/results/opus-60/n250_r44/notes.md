# Notes

## Data
250 rows, x0 in (0,1), y in (-1.8, 2.6). Scatter shows a smooth wavy curve (min near x=0.25,
max near x=0.65, falling to -0.9 at x=1) plus a second, steep, nearly linear branch that
splits off around x≈0.6 and reaches 2.6 at x=1. Noise on the lower branch shrinks with x
(sd about 0.35 down to about 0.1). Within-branch residuals look Gaussian (slightly light-tailed), so I used Normal components.

## Model family
Two-component Gaussian mixture:
- lower: polynomial mean (degree A), log-sd polynomial (degree B)
- upper: polynomial mean (degree C), log-sd polynomial (degree D)
- gate: weight of the upper component = sigmoid(polynomial of x, degree G)
Fitted by maximum likelihood (BFGS, several random restarts).

## Validation
10-fold CV repeated twice (different shuffles), refitting on every fold; the score is the mean held-out NLL.
Selected results (A,B,C,D,G): train / CV
- (3,1,1,1,1) 0.169 / 0.224
- (3,1,1,0,1) 0.185 / 0.246
- (4,1,1,1,1) 0.141 / 0.201
- (4,2,1,1,1) 0.134 / 0.199
- (4,1,2,1,1) 0.130 / 0.193
- (4,2,2,1,1) 0.125 / **0.1905**  <- chosen
- (5,2,2,1,1) 0.125 / 0.197
- (4,2,2,1,2) 0.120 / 0.220
- (4,1,1,1,2) 0.134 / 0.208
I also tried a "sharp" gate pmax*sigmoid(k(x-c)) to model the bifurcation directly. It was
consistently worse in CV (0.218-0.228), probably because the fitted switch point is unstable across folds.

## Final
(4,2,2,1,1): 15 constants, fitted on all data, train NLL 0.125, CV NLL 0.1905. Fold-to-fold
noise is about ±0.02. A refit on the full data should help slightly, while picking the
best of ~15 configurations adds some optimism; I estimate about 0.19.

expected_nll: 0.19
