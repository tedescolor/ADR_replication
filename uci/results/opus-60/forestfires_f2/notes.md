# Notes

## Data
- y: ~45% of rows sit in a blurred spike around 0 (values from -1.4 to ~1, looks like an exact
  zero plus noise with sd ~0.6); the rest is a positive, very right-skewed tail (up to ~1090).
- Covariates have tiny correlations with log(1+y) (|r| < 0.1).

## Validation
5-fold CV repeated 3 times (10 times with paired per-row differences for the covariate
comparisons); every fit is ML (L-BFGS-B with restarts) on the training folds only.

## Candidates (CV mean NLL)
- Normal(zero spike) + LogNormal: **2.903** (10-rep CV: 2.905); in-sample 2.892
- StudentT(3) spike + LogNormal: 2.916; Laplace spike + LogNormal: 2.918; SkewNormal spike: 2.903 (extra parameter, no gain)
- Normal + Normal + LogNormal: 2.911; Normal + 2 LogNormals: 2.904-2.907
- Covariates (linear in standardized x, on the spike weight logit and/or LogNormal mu), one at a
  time, plus derived features (month cycle, squares, log transforms, rain indicator): best
  single gain ~0.005; best combo (x0 on weight, x9,x10 on mu) improves on the baseline by
  0.005 ± 0.0085 (paired SE), i.e. not significant. Rain (x11) gets heavily worse because of
  outliers.

## Choice
Unconditional 2-component mixture: 0.446 Normal(-0.091, 0.586) + 0.554 LogNormal(1.677, 1.695),
fit on all the data. 6 constants. The covariates gave no reliable improvement, and the simpler
program is more robust everywhere in the covariate box.

Expected NLL: the CV estimate is ~2.905. The heavy right tail makes the held-out mean noisy, so I
round it up a little.

expected_nll: 2.91
