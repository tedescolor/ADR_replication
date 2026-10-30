# Notes

## Data
1000 rows, x0 in [0,1]. Scatter: a single noisy curve (sd ~0.3) for x0 < ~0.55, which then
splits into two branches: an upper branch rising almost linearly to ~2.5 at x0=1 (sd ~0.1),
and a lower branch peaking near 0.1 at x0~0.65 and falling to ~-0.9 at x0=1. The split is
roughly 50/50 after the bifurcation. The noise looks close to Gaussian.

## Model family
2-component Gaussian mixture. Each component has a polynomial mean in x0, a log-sd
polynomial and a shared logit-polynomial weight. Component 1 is the trunk plus the upper
branch, and component 2 is the lower branch. The weight goes to ~1 on component 1 for small
x0 and to ~0.5 after the split. Fitted by MLE (BFGS, several jittered starts from a
hand-made initialisation of the two branches).

## Validation
5-fold CV, repeated with 2 to 4 different random partitions, refitting on each training
fold. Configurations are given as (number of coefficients for mean1, mean2, logsd1, logsd2,
logit w):
- (5,4,3,3,3): 0.2405; (6,4,3,3,3): 0.2409; (7,5,4,3,4): 0.2447 (larger means overfit or hit bad optima)
- (5,4,2,2,3): 0.2288; (4,4,2,2,3): 0.2299; (5,3,2,2,3): 0.2297
- **(4,3,2,2,3): 0.2282** (14 constants, train 0.2093)
- (3,3,2,2,3): 0.42; (4,2,2,2,3): 0.29 (too rigid)
- logsd degree 0 or weight degree 1 are worse (0.236 to 0.258)
- Student-t components with a shared df (df about 22): CV 0.2272, a gain of only 0.001, so I
  did not use them.

## Final
Configuration (4,3,2,2,3): cubic mean for the trunk/upper branch, quadratic mean for the
lower branch, log-linear sds, quadratic logit weight. 14 numeric constants plus the 1.0 in
1-w. It has the best CV score and is the simplest of the good models. Refit on all data.
The CV estimate is 0.228. I add a little for selection optimism.

expected_nll: 0.232
