# Notes

## Data findings
- y is in roughly [0.9497, 1.0004]. It is nearly uniform over a 0.001 grid (0.950..1.000), with small jitter (sd ≈ 1.5e-4).
- x0 takes 9 well-separated levels (cluster sd ≈ 0.026). `int(x0 - 0.7)` maps the whole domain box onto levels 0..8.
- The covariates are all highly collinear functions of the level, and marginal correlations with y are about 0. The y signal sits in small within-level deviations.
- A compressor-efficiency-like feature, (PR^0.2857 - 1)/(x9 - x8) with PR = x12/x11, correlates with y at 0.75–0.98 within a level. The constant-free version log(x12/x11)/(x9 - x8) does exactly as well.
- Residual sd depends strongly on the level: about 0.0044 at level 0 and 0.0015 at level 8. Standardised residuals are close to Gaussian (kurtosis 0.14, skew 0). Student-t did not help.

## What I tried (holdout / 5-fold CV)
- Global linear model: residual sd 0.0075. Global quadratic: 0.0037.
- Greedy forward selection over z_i * L^p and z_i * z_j terms (smooth in x0): needs more than 35 terms to reach about 0.0032.
- Per-level linear model on all features: sd 0.0029, which looks like the floor (per-level quadratic is no better). Per-level greedy selection order: e, x9, x4, x10, x13, x2, x3, x12, x7.
- Coefficients made smooth (polynomial) in x0: about 10% worse sd than per-level tables at a similar constant count.
- Final structure: per-level intercept and slopes on (e, x9, x4, x10, x13, x2), a globally shared slope on x7, and a per-level sd. A shared x7 slope beat adding any 7th per-level feature.
- Bounded support: a truncated normal on [0.9495, 1.0005] would gain 0.066 nats, but no truncated family is available. Clamping the mean to [0.9515, 0.9985] gets about 0.02 of that for 2 constants. Those 2 constants came from using the constant-free log efficiency feature.

## Validation
5-fold CV (random folds; the full per-level fit is redone in each fold):
- per-level 6 features + per-level sd: −4.434
- + shared x7: −4.468 (seeds 0/1/2: −4.468, −4.467, −4.467)
- + mean clamp (final): −4.4886
- in-sample for the final fit: −4.4963, so the optimism is about 0.008

## Final program
80 constants: a 9×8 table (intercept, 6 slopes and sd for each level), plus 1 shared slope, the 0.7 level offset, 2 index integers (k//3, k%3) and 2 clamp bounds. Passes the contract check.

expected_nll: -4.485
