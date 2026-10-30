# Notes
Data: 1 covariate x0 in [0,1]; mean of y is non-monotone (dip near 0.25, rise to ~1 at x=1); sd is U-shaped (high at 0, min ~0.45, grows to ~1.5 at x=1). Residuals roughly symmetric.
Validation: 10-fold CV (shuffled), mean NLL, parameters refit by MLE in each fold.
Tried Normal vs StudentT with polynomial mean (deg 3-7) and polynomial log-sd (deg 1-6). StudentT gave no gain (df large). Mean deg 4 was best; log-sd deg 4->6 gave 0.612->0.610 (tiny), deg 1 was bad (0.75).
Final: Normal(poly4(x), exp(poly5(x))), 11 coefficients, fit on all data. CV NLL 0.6112 (train 0.602).
expected_nll: 0.612
