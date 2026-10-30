# Notes

## Data
250 rows, x0 in (0,1). The scatter plot shows two crossing curves. One is S-shaped, rising from
about -0.8 at x=0.25 to 2.4 at x=1, with tight noise on the right. The other is a hump: it peaks
near 0 at x≈0.68 and falls to about -1 at x=1. For x<0.5 the two curves overlap, so the data
there looks like one noisy band.

## Candidates (repeated 10-fold CV: 2 repeats, 6 random restarts per fit; mean held-out NLL)
Spec = (mean poly coeffs comp1, log-sd coeffs comp1, mean comp2, log-sd comp2, gate logit coeffs)
- single normal, cubic mean, log-linear sd: 0.747
- 2-normal mixture, cubic means, log-linear sds, linear gate: 0.280
- same, constant gate (4,2,4,2,1): **0.273**
- quintic means, constant gate: 0.270 (differences of about 0.01 are within CV/optimizer noise)
- quartic/cubic means: 0.286; sextic: 0.295; quadratic log-sd: 0.282-0.291; constant sds: 0.342
- Student-t first component: 0.281
- quadratic gate variants: 0.29-0.33

## Choice
Mixture of 2 normals with cubic means, log-linear sds and a constant weight (15 constants).
It is the simplest model within noise of the best CV score; adding flexibility makes CV
worse. Fitted on all the data by ML (BFGS, 20 restarts); train NLL is 0.210.
The CV estimate of about 0.27 comes from models trained on 90% of the data. Allowing for
small-sample optimism, I estimate:

expected_nll: 0.28
