# Notes
Data: 1 covariate x0 in [0,1]; y has a nonlinear (dip then rise) mean and strongly heteroscedastic sd (min ~0.27 near x=0.45, ~1.5 near x=1).
Validation: 5-fold CV (shuffled, seed 0), MLE refit per fold. Compared Normal / StudentT / SkewNormal, polynomial mean degree 3-7, log-sd polynomial degree 2-4, plus sd forms (const+exp, power).
Results (CV NLL): Normal, mean deg 4, log-sd deg 4: 0.606 (best). log-sd deg 2-3: 0.625. const+exp sd: 0.634. StudentT/SkewNormal gave no gain (tails are Gaussian). Higher mean degrees were slightly worse.
Final: Normal(poly4(x), exp(poly4(x))) fit on all 1000 rows; 10 constants. Train NLL 0.599.
expected_nll: 0.607
