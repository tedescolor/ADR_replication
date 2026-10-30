# Notes
Data: 1 covariate x0 in [0,1]; y has a nonlinear mean in x0 and a U-shaped, heteroscedastic spread (small near 0.5, large near 1).
Validation: 5-fold CV repeated 3x (different shuffles), parametric fits by ML (BFGS).
Grid: mean polynomial degree 3-6, log-sd polynomial degree 1-3, noise Normal / StudentT / SkewNormal.
Results (CV NLL): linear log-sd ~0.75; quadratic log-sd ~0.621-0.622 for all families and mean degrees 3-6. Heavier tails or skew gave no gain.
Final: Normal, quartic mean, exp(quadratic) sd (8 constants + structure), fit on all data (train NLL 0.613).
The CV differences between the top choices are within noise, so the simplest was chosen.
expected_nll: 0.622
