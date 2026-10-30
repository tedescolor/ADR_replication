# Notes

## Data
250 rows, x0 in (0,1). Scatter shows a single noisy curve for x < ~0.61 (U-shaped dip to
-0.75 at x~0.22, sd ~0.25-0.35, larger at small x), which bifurcates at x ~0.61 into two
narrow branches (sd ~0.12): a rising one (to 2.6 at x=1) and a falling concave one (to -0.9),
each holding ~half of the points.

## Model family
Two-component Normal mixture:
- comp 1 (main curve, then upper branch): mean = cubic in u=x-0.5; log sd linear in u plus a
  step at the split (narrower after bifurcation);
- comp 2 (lower branch): mean quadratic in u, constant sd;
- weight of comp 2 = 0.5 * g, g = sigmoid(200 (x - c)) a near-step gate at c ~0.613.
Fitted by maximum likelihood (BFGS, several starts).

## Validation
10-fold CV repeated twice (different shuffles), refitting everything inside each fold.
Mean held-out NLL:
- free gate steepness, quartic mean1, sd1 linear+step:     0.245 (train 0.163)
- same, homoscedastic sd1:                                 0.254
- quintic / quadratic-in-sd / linear lower branch: worse (0.25-0.31)
- fixed steep gate (k=200), cubic mean1, Normal:           0.228
- same with Student-t components (shared df, df~18):       0.232
- quartic + t:                                             0.236
- + broad background outlier component: weight -> 0, same CV
- fixed 0.5 branch split (final, 12 fitted constants):     0.2248 (train 0.168)
Held-out loss is dominated by a few outliers (e.g. a point between the branches at x~0.78);
Heavier tails did not pay off in CV.

## Final
model.py: the fixed-split, fixed-gate cubic/quadratic Normal mixture, refit on all data.
Full-data NLL 0.168; CV 0.225. Given CV noise from outliers, estimate slightly above CV.

expected_nll: 0.23
