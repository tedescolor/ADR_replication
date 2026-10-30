# Notes
Data: 250 rows, one covariate x0 in [0,1]. Mean of y is nonlinear (dip near 0.2, rise after 0.5); spread is heteroscedastic (small mid-range, large near x=1).
Validation: 10-fold CV, repeated 3-5 times with different shuffles; NLL on held-out folds, parameters refit in each fold.
Candidates: Normal vs StudentT; polynomial mean degree 2-6; log-sd polynomial degree 1-3.
CV NLL: mean deg 3 + log-sd deg 2 Normal = 0.648 (best; deg 2 mean 0.739, deg 4+ 0.649-0.658, log-sd deg 1 0.77). StudentT gave no gain.
Final: Normal(cubic(x), exp(quadratic(x))), 7 constants, fit by max likelihood on all data.
The CV estimate is mildly optimistic since the structure was picked by CV, so I added a small margin.

expected_nll: 0.66
