# Notes

## Data
1000 rows, x0 in (0,1). Scatter: one noisy curve for x<~0.55 (mean dips to about -0.8 at x=0.22),
which splits into two branches for x>0.6: an upper, roughly linear one (up to 2.5 at x=1) and a
lower one curving down to -1 at x=1. Noise shrinks steadily with x (sd ~0.4 at x=0, ~0.1 at x=1).
Within each branch the residuals look Gaussian: a Student-t fit sent df to about 1800, so Gaussian.

## Model family
A 2-component Gaussian mixture whose parameters depend on x: polynomial means, a scale per
component, and a logistic weight. Fitted by MLE (BFGS / L-BFGS-B, started from a preliminary fit).

## Validation
5-fold CV repeated over 2 random partitions (10 held-out folds), refitting on each training part.
Mean held-out NLL (fold se ~0.02, but paired differences between models are much smaller):

| variant | #par | CV NLL |
|---|---|---|
| free polys: cubic means, linear log-sd, linear logit w | 14 | 0.2491 (single 5-fold) |
| quartic/quintic/sextic means, more sd/w terms | 16-23 | 0.250-0.254 |
| cubic means, shared linear log-sd, const w (A) | 11 | 0.2523 |
| A with a cubic coefficient shared by both means (C) | 10 | 0.2516 |
| C + linear logit weight (D) | 11 | 0.2472 |
| D with Student-t noise | 12 | 0.2472 |
| **D with sd linear in x on the raw scale, not the log scale (H)** | **11** | **0.2445** |
| H + quadratic sd / separate sds / separate cubics / quartic means | 12-14 | 0.2449-0.2470 |
| H with constant weight | 10 | 0.2489 |

H gives the lowest CV. Every extension of it was flat or worse. PIT values of the final model
are uniform within each of 8 x-bins (all KS p-values > 0.4), so no structure is left to model.

## Final program (H, 11 constants)
m1 = 0.0162 - 7.9747x + 21.1432x^2 - 10.7132x^3
m2 = -0.3624 - 3.7595x + 13.845x^2 - 10.7132x^3
sd = 0.3915 - 0.2963x (shared by both components)
w = sigmoid(1.8587 - 2.1704x) on component 1
Training NLL 0.2309. CV NLL 0.2445.

## Process note
Early on I accidentally ran 4 fitting processes in parallel for about 2 minutes, which exceeded
the 2-process limit. Every run after that used at most 2 processes.

expected_nll: 0.245
