# Notes

## Data
1000 rows, x0 in (0,1). Scatter: one noisy curve for x < ~0.5, which then splits into
two branches. The upper branch continues the left curve smoothly up to y≈2.5 at x=1. The lower
branch bends down to y≈-1 at x=1. About half the points (0.53) fall on each branch. The noise
is Gaussian-looking (residual skew and kurtosis ≈ 0) and shrinks with x: sd ≈ 0.4 near x=0
and ≈ 0.11 near x=1. Log-linear in x fits it well.

## Validation
5-fold CV repeated twice (10 fits per config), mean held-out NLL with natural log. All fits
are maximum likelihood with BFGS. To avoid bad local optima I used multiple starts: a grid
over the split point and power, plus a warm start. Fold-level SE ≈ 0.016, but paired
differences between configs are much smaller than that.

## Tried (CV NLL)
- Free 2-component Gaussian mixture: polynomial means (deg 3–7), polynomial log-sds, logistic
  weight. Best ≈ 0.161–0.162 with about 20 parameters.
- Symmetric split: base(x) ± relu(x-c)^p * k, shared log-linear sd, w=0.5. 0.158–0.159 with
  11–12 parameters. A free weight did not help (0.160). A quadratic log-sd gave ≈0.001.
- Asymmetric "upper = smooth polynomial, lower = upper − relu(x−c)^p·(k0+k1 t)", shared
  log-linear sd, w = 0.5:
  (4,0,1) 0.1574, (5,0,1) 0.1561, (4,1,1) 0.1557, **(5,1,1) 0.1548**, (6,1,1) 0.1570,
  (5,1,2) 0.1544.
- Calibration check of the structured fit: PIT KS p = 0.999. Standardized residuals by
  x-decile have sd ≈ 1.

## Final
The (5,1,1) asymmetric model, an equal-weight mixture of two Normals:
- upper mean = quintic in t = 2x−1;
- lower mean = upper − max(x−0.519, 0)^0.927 · (3.07 + 3.83 t);
- shared sd = exp(−1.552 − 0.691 t).

It has 12 fitted constants. The contract counts 21 literals (including exponents and 0.5
weights) and 154 AST nodes. I picked it because it was essentially
tied for the best CV score (the 13-parameter (5,1,2) was only 0.0004 better) and it is simple.
Training NLL is 0.1429 and CV NLL is 0.1548. The final fit uses all the data, so I expect
held-out performance close to the CV value.

expected_nll: 0.155
