# Notes

## Data
85 rows, y > 0 (0.006 .. 1.09), right-skewed. Covariates look like a designed experiment
(x0 ~ {10,15,18,20,22,26,28,30}, x1 ~ {12..25}, x2 ~ {100,200,300}, x3 ~ 15..100) plus a
continuous x4. y correlates strongly with x4 (+0.72) and x2 (-0.62).

## Validation
Repeated 5-fold CV (4-6 random partitions, same seeds for all candidates), mean held-out NLL
on the original y scale; LOO for diagnostics. Fold-to-fold spread ~0.02-0.03, so differences
below ~0.02 are treated as noise.

## What I tried (CV NLL)
- Normal, linear / log features, with or without heteroscedastic log-scale: -0.43 .. -0.72
  (scale models overfit).
- LogNormal, linear features: -0.90. (sqrt-y normal ~ -0.99 but not expressible in adr.dist.)
- Gamma, log link, all linear features, constant shape: -0.98/-0.99. Sqrt link -0.99 but
  less stable; identity link and SkewNormal unstable; shape depending on mean: worse.
- Extra features (squares, interactions, log x2, dropping x0/x1): no reliable gain.
- LOO showed a few points dominate (y=0.006 at x2=300, x1=25 cost 6.3 nats). Adding a fixed
  contamination component Gamma(shape 2, same mean) with weight 0.1 gave -1.051
  (grid over w in 0.03..0.2, a2 in 1..4 all -1.046..-1.052; freeing w, a2 overfits: -0.89).
- With that mixture, using log(x4) instead of x4: **-1.066 ± 0.033** (chosen).

## Final program
mean m = exp(b0 + b0..b3 * x0..x3 + b4 * log x4); y ~ 0.9 Gamma(shape 12.46, mean m)
+ 0.1 Gamma(shape 2, mean m). 11 constants, fit by MLE on all 85 rows (train NLL -1.166).
Chosen because it is the best CV score, uses few parameters, is valid over the whole
covariate box (log-link mean always positive, x4 > 0), and the heavy component protects
against tail surprises in held-out data.

Expected NLL: CV -1.066, shrunk a bit for selection among ~20 variants.

expected_nll: -1.04
