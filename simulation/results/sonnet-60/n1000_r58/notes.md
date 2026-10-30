# Notes
Data: 1 covariate x0 in [0,1]; mean of y is nonlinear (dip then rise), spread varies strongly with x0 (small near 0.5, large near 1).
Tried: Normal and StudentT, polynomial mean (deg 2-7) and polynomial log-sd (deg 1-6), 10-fold CV (shuffled).
Normal ~ StudentT (no heavy-tail gain). Mean deg 3 is enough (deg 2 much worse, >3 no gain); log-sd deg 4 gave 0.6187 CV, deg 6 only 0.616 (marginal), so kept deg 4.
Final: Normal(cubic mean, exp(quartic)) fit by ML on all data; 9 constants for the mean/sd coefficients plus the 0.5 centering (10 total).
Train NLL 0.610, 10-fold CV NLL 0.619.
expected_nll: 0.62
