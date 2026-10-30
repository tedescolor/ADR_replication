# Notes

## Data
85 rows; covariates look like a designed experiment (x0 ~ {10,15,20,30}, x1 ~ {15,17,20,...},
x2 ~ {100,200,300}, x3 ~ 15..100) plus a continuous x4 correlated with them. y > 0, right-skewed;
log y is roughly linear in the covariates. Group means of log y are linear in raw x2
(-0.96, -1.80, -2.64 at 100/200/300), so x2 enters on the raw scale; the others on log scale.
One very low point (y = 0.006) dominates the tail behaviour.

## Validation
Leave-one-out and 10x repeated 5-fold CV of mean NLL on the original y scale (log-scale
densities corrected by the -log y Jacobian). Rep-to-rep std of 5-fold CV ~0.03-0.1, so
differences < ~0.03 are treated as noise.

## Tried (LOO / CV mean NLL)
- Normal on log y, all log covariates: ~ -0.92; Student-t on log y: ~ -1.02 to -1.07 (df ~ 4).
- Skew-normal on log y: unstable/worse. 2-component normal mixtures on log y: worse (-0.87..-1.03).
- Heteroscedastic scale (on x0, x1, x2, x3 or x4): never helped (worse or equal).
- Quadratic/interaction terms: no gain. x3 adds nothing once x4 is in.
- Raw x2 instead of log x2: clear gain. log-t: LOO -1.119; Gamma GLM: LOO -1.117;
  lognormal scale-mixture approx of log-t: -1.106; 50/50 average of Gamma and log-t: -1.119.

## Final
Gamma(shape, rate = shape/mean), log mean = b0 + b1 log x0 + b2 log x1 + b3 x2 + b4 log x4,
constant shape (8.56). 6 fitted constants (7 literals). Chosen as the simplest model tied for
best and naturally expressible with adr.dist. LOO -1.117, 10x5-fold CV -1.098 (sd 0.03),
in-sample -1.222. Held-out estimate taken slightly conservative of LOO given the heavy left
tail (occasional very small y) and small n.

expected_nll: -1.08
