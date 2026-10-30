# Notes

## Data
250 rows, x0 in [0,1], y in [-1.6, 2.5]. Scatter plot (work/scatter.png): for x0 < ~0.55 a
single noisy curve (sd ~0.3 near x=0, ~0.2 near x=0.5) with a minimum near x=0.25. Around
x0 ~ 0.6 the data split into two branches: an upper, tight (sd ~0.14) roughly linear branch
rising to 2.5 at x=1, and a lower branch that keeps following the main curve down to -1 at x=1.
Roughly half the points for x > 0.6 are on each branch.

## Model family
A 2-component Gaussian mixture, with z = 2*x0 - 1:
- main component: polynomial mean in z (degree da), log-sd polynomial (degree ds);
- upper branch: polynomial mean (degree db), constant sd;
- weight of the upper branch: cap * sigmoid(w0 + w1*z).
Fitted by maximum likelihood (BFGS, 3 initialisations from a hand split of the branches).

## Validation
10-fold CV repeated twice (20 fits per configuration), mean held-out NLL:
- uncapped sigmoid gate, (da,ds,db) = (6,2,2): 0.286; (6,2,1): 0.294; (5,2,1): 0.330
- Student-t main component: no gain (0.285)
- capped gate (cap ~0.47): (6,2,2) 0.254; (6,1,2) 0.254; (6,3,2) 0.255; (6,2,3) 0.255;
  (7,2,2) 0.260; (8,2,2) 0.269; (5,2,2) 0.310; quadratic log-sd for the upper branch: 0.266
- With the free gate, the full-data fit turned the gate into a hard step at x=0.42, which
  is fragile. Fixing the gate slope w1 (on z): 10 -> 0.260, 20 -> 0.2545, 40 -> 0.266,
  80 -> 0.269.

## Final choice
(6,2,2) with a capped gate and fixed slope 20 (17 fitted values; the program has 22 numeric literals and 178 AST nodes).
It ties for the best CV score (0.2545) while keeping a smooth gate (centre x0 ~ 0.44, cap
0.477). Training NLL 0.179. The CV score comes from fits on 90% of the data. Fitting on the
full data should help a little, and choosing among ~15 configurations should hurt a little. I
round the estimate up slightly.

expected_nll: 0.26
