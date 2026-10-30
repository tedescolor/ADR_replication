# Notes

## Data
413 rows, 12 covariates, scalar y. y is a lump around 0 (sd ~0.6, about 24% of the rows are negative,
down to -1.37) plus a very heavy right tail (median 0.9, 90% quantile 22, max 211).
The covariates look like grid coordinates, month, day of week and continuous weather-type indices.
x11 has one extreme outlier.

## Validation
Repeated K-fold CV (10 folds x 2-3 repetitions) with the mean held-out NLL. Every fold is refit by
maximum likelihood with multi-start BFGS / Nelder-Mead.

## What I tried
- Unconditional mixtures (CV NLL):
  - Normal+LogNormal: **2.823**
  - StudentT+LogNormal: 2.823 (df went to infinity, so it is the same model)
  - SkewNormal+LogNormal: 2.825
  - Normal+Normal+LogNormal: 2.830
  - Normal+SkewNormal+LogNormal: 2.831
  - Normal+LogNormal+LogNormal: 2.834
  - SkewNormal+LogNormal+LogNormal: 2.838
  - Laplace+LogNormal: 2.842
  - Normal+Gamma: 2.876
- Covariate effects on top of Normal+LogNormal. I added one covariate at a time, either linearly in
  the mixture logit, in the lognormal location or in the lognormal log-scale. All 12 raw covariates
  were tried, plus non-linear features: month indicators, weekend, rain>0, log of x5 and x7,
  thresholds, a squared temperature term and log(x9). No change beat about 0.005 nats, which is far
  below the CV standard error (around 0.09 per-point SD/sqrt(n)). Several changes hurt; the x11
  outlier hurt badly. I found no usable signal, so the final model ignores x.

## Final program
Mixture(0.444 Normal(-0.054, 0.607), 0.556 LogNormal(mu=1.572, sigma=1.590)), fit by MLE on all
the data. It has 6 constants.
- Train NLL: 2.806
- 10-fold CV NLL: 2.823

The held-out mean NLL is dominated by tail rows, so I expect a standard error of about 0.1 on a
test set of similar size. I add a little pessimism for possible extreme values in the tail.

expected_nll: 2.84
