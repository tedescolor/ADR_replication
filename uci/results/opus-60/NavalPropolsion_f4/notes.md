# Notes

## Data findings
- `y` lies on a grid of step 0.001 (51 levels, 0.950 ... 1.000, roughly uniform), plus small
  Gaussian noise (sd about 1.46e-4, the same for every level and every operating point).
- `x1/3` rounds exactly to an integer 1..9 (9 operating points; `x0` gives the same grouping).
  Inside each group the covariates are noisy, and `y` is predicted linearly with residual sd
  of about 2.9e-3 (about 3 grid levels). The residual sd falls from about 4e-3 in group 1 to
  1.4e-3 in group 9. Linear correlations over the whole dataset are near zero, so per-group
  structure is essential.
- The most useful features (greedy per-group selection) are x9, x12, x4, x10, x3, x2, x7, x13.
  Adding quadratic terms gave almost nothing.

## Model
- `i = round(x1/3) - 1`; mean `z = T[i] + sum_j x_j * (a_j + b_j * i)`. T is a table of 9
  per-group intercepts. Final features: x9, x12, x4, x10, x3, x2, x13, x7 with linear-in-i slopes,
  plus x15 and x14 with constant slopes.
- Envelope sd `sg = exp(quadratic in i)`.
- Density: a "comb". It is a mixture of 25 Normals centred on the grid levels nearest to z,
  with the window kept inside [0.950, 1.000]. Each has sd s = 1.46e-4, and the weights are
  proportional to exp(-(c-z)^2 / (2 sg^2)), i.e. a discretised Gaussian over levels truncated
  to the support. The comb is built as a Mixture of 5 Mixtures of 5. It is mixed with weight
  0.002 with a broad Normal(z, 2 sg), which guards against levels outside the 25-level window.
- All parameters were fitted jointly by maximum likelihood (exact comb likelihood over all 51
  levels, L-BFGS with analytic gradient), starting from least squares.
- 50 constants, 473 AST nodes.

## Validation
- Feature and degree selection: 5-fold CV RMSE of linear designs (group dummies + x_j * i^p).
- NLL: holdout (fit on 80%, evaluate on the remaining 20%, 2 different folds) with the exact
  comb likelihood. Full-data training NLL is -4.943. The evaluated model.py over every 5th row
  gives -4.920 (the 0.002 fallback weight and 25-level truncation cost about 0.003).
- Reference: a plain Gaussian with the same mean/sd would score about -4.4. The comb gains
  about 0.5 nats from the discreteness of y.

## Candidates (2 x 80/20 holdout, exact comb NLL)
- 7 features (x4 constant slope), 26 fitted params: holdout -4.900 / -4.903 (mean -4.901).
- 10 features (final), 31 fitted params: holdout -4.919 / -4.926 (mean -4.922). Chosen.
- The train-to-holdout gap is about 0.025. The 25-level window and the 0.002 fallback cost
  about 0.003 more, so I expect about -4.92 on new data and report a slightly conservative
  number.

expected_nll: -4.91
