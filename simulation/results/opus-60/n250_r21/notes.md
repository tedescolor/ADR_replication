# Notes

## Data
250 rows, one covariate x0 in [0,1]. A scatter plot shows two branches:
- a lower, wavy, fairly noisy branch (min ~-0.8 near x=0.25, max ~0.2 near x=0.65, then down to ~-0.8 at x=1),
  with noise sd ~0.35 at low x that shrinks at high x;
- an upper, tight (sd ~0.17), roughly linear branch (y ~ 1.62 + 5.07 (x-0.8)), which crosses the lower
  branch around x~0.42 and carries ~57% of the mass beyond that point.

## Model family
A 2-component Normal mixture with an x-dependent weight, fit by maximum likelihood (BFGS, multiple starts).

## Validation
5-fold CV repeated 3 times (different splits), refitting everything (including the threshold) per fold;
mean held-out NLL per row:
- lower sinusoid mean, logistic weight sigmoid(a+bx): 0.335
- lower poly4 mean, log-quadratic sd, logistic weight: 0.321
- logistic weight replaced by capped step cap*sigmoid(k(x-thr)), free k: poly4 0.312, cubic 0.313, poly5 0.54 (unstable),
  sinusoid lower 0.336, quadratic upper mean 0.318, log-linear upper sd 0.317, log-linear lower sd 0.314
- fixing k=300 for a hard step: 0.321 (less stable); Student-t upper component: 0.326
- **cubic lower mean, log-quadratic lower sd, linear upper mean, constant upper sd, weight
  cap*sigmoid(50(x-thr)): 0.312 (sd across repeats 0.004)**. This is the final model.
Profiling the threshold on the full data puts a clear optimum at ~0.42 (0.63 is a worse local optimum).

## Final program
12 fitted parameters (16 literals). Training NLL 0.266; CV NLL 0.312.
Fitting on all 250 rows rather than 200 should help slightly, but the step location is still uncertain, so
I take the CV figure, rounded up a little, as the estimate.

expected_nll: 0.315
