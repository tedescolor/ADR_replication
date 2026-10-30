# Notes

## Data
- 9547 rows, 16 covariates, y in [0.9497, 1.0004], essentially **uniform on [0.95, 1.0]**.
- Almost every covariate is a smooth function of x0 (R^2 > 0.96 on a polynomial in x0);
  x8 and x11 are pure noise around constants. y is only visible in the small deviations of
  the other covariates from their x0-trend, and how they map to y depends on x0.
  Marginal linear correlations with y are all < 0.06.
- Polynomial regressions (holdout residual sd): linear 0.0074, full quadratic 0.0036,
  full cubic (969 terms) 0.0031. Greedy varying-coefficient models (x_j * u^k with u = x0 - 5,
  plus a few products x_i*x_j) reach about 0.0030 with 60 terms. That looks like the noise floor.
- The residual sd depends strongly on x0 (0.0037 at low x0, 0.0016 at high x0), so the
  log-scale is modelled as quadratic in u.
- Because y is bounded and uniform, the ideal predictive density is close to
  normal(t, s) **truncated** to [0.95, 1.0]. A hard truncated normal improved holdout NLL from
  -4.43 (SkewNormal with mean-dependent skew) to -4.47. Hard truncation isn't available and
  would be dangerous, so I approximate it with a SkewNormal. Its location offset, scale and
  shape are smooth functions of d, the distance from t to the nearest edge in units of s.
  The functional forms came from KL-fitting a skew-normal to truncated normals.
  The edges stay soft: a y 0.001 beyond the observed range still gets a finite, moderate log-density.

## Validation
- Random split 7000 train / 2547 holdout (seed 0). All model selection (greedy term order,
  block choice) used the training part only. Parameters were fitted by maximum likelihood
  (L-BFGS, mean in QR-orthogonalised coordinates, then mapped back to raw coefficients).
- Holdout NLL of the candidates:
  - Normal, 30 greedy terms, log-sigma quadratic: -4.404
  - SkewNormal with skew a cubic function of the mean: -4.431
  - edge-aware SkewNormal (truncated-normal approximation), 30 terms: -4.471; 60 terms: -4.505.
    The 60-term version broke the 500 AST node limit (783 nodes).
  - Restructured to fit the limits: t is Horner in u, and each level is a few
    `dot(coefs, tuple)` blocks over Xa = x1..x6, Xb = x12..x15, Xd = x7..x10, and
    Xc = (x9^2, x7^2, x12^2, x4^2). The blocks were chosen by OLS gain per literal and node
    (backward elimination). The edge part shares one term w = d*exp(-k d^2) between the
    offset and the scale (costs only 0.001 nats).
  - final structure (v8b): holdout NLL **-4.496** (seed-0 split), **-4.489** (seed-1 split,
    refitted from scratch). The train/holdout gap is about 0, which is unsurprising with
    ~70 parameters and 7000 rows.
- The final program is refitted on all 9547 rows (in-sample NLL -4.490). The worst
  per-row log-density in-sample is -4.7, so the tails are well behaved.

## Final program
- 58 mean coefficients, 3 for log-scale, 7 for the edge/shape part, plus the centring and
  clamp constants: 69 literals, 499 AST nodes. Passes `adr.contract` with the domain check.
  The shape argument is clamped (d >= -3), so the program stays finite at extreme covariates.

## Estimate
The average of the two holdout splits is -4.49. Fitting on all rows should help slightly.

expected_nll: -4.49
