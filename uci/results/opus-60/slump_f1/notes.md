# Notes

## Data exploration
- 82 rows, 9 covariates. x0..x6 sum to an almost constant total (~2290, sd 40); x7 and x8 are strongly correlated (0.9).
- Linear Gaussian model: LOO NLL about 2.42 (residual sd about 2.3).
- ARD Gaussian process (RBF kernel): honest 10-fold NLL about 0.50, noise sd about 0.2. Its length scales show that y is a smooth,
  almost noise-free function of x0..x6. x7 and x8 are irrelevant, and x4 and x5 matter only weakly. GP profiles: x3 is strongly
  nonlinear (non-monotone with a cubic-like shape) and interacts with x0, x1, x2; x6 is concave and interacts with x0.
  A GP cannot be written within 80 constants, so it served only as a reference for what is achievable.

## Candidates (validation = LOO and 5x repeated 10-fold CV; sigma from training residuals; nested CV where structure was selected)
- linear, best subsets: about 2.42
- quadratic in all 9 covariates, ridge: about 1.20
- quadratic in x0..x6 (36 terms): LOO about 1.05-1.12; dropping x4/x5 terms makes it worse (1.39)
- ratio-based parametric (Abrams-like x3/(x0+k1 x1+k2 x2) with a cubic, etc.): LOO 1.84 (worse)
- full cubic in the main variables: overfits (1.6-2.9)
- forward selection of cubic terms on top of the quadratic: nested CV shows it HURTS (selection is unstable)
- tanh MLP with linear skip (3-6 hidden units): 10-fold about 0.98-1.5, unstable
- quadratic in x0..x6 + x3^3 (the cubic term suggested a priori by the GP profile), ridge on standardized features:
  repeated 10-fold NLL: lambda=0.01 1.00, lambda=0.1 0.972, lambda=0.3 1.005 (Student-t df=10 with scale 1.2 x residual sd)
  (Normal: 0.974; t5: 0.978)

## Final program
Mean = full quadratic polynomial in d_j = x_j - c_j (j=0..6) plus a d3^3 term, fit by ridge (lambda=0.1 on standardized
features, then converted exactly to raw-centered coefficients). Noise = StudentT(df=10, scale=1.2 x residual sd 0.471 = 0.565).
The t-distribution and the 1.2 inflation cover CV misfit in a few rows and the extra uncertainty from extrapolating to new points.
It uses 7 centers + 1 + 7 + 28 + 1 coefficients + 2 noise parameters = 46 constants. Training NLL 0.59; CV NLL 0.97.
The estimate below is set slightly above the CV value because the x3^3 term, lambda and df were chosen on the same data.

expected_nll: 1.02
