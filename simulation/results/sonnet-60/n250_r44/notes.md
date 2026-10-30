# Notes
Data: 1 covariate x0 in [0,1]; mean of y is nonlinear (dip near 0.25, rise after 0.5); spread is U-shaped in x0 (min near 0.55, large near 1).
Validation: 10-fold CV, 3 repeats, mean held-out NLL. Candidates: Normal / StudentT, polynomial mean degree 3-6, polynomial log-sd degree 1-4.
- log-sd linear: ~0.74; log-sd quadratic: ~0.60 (big gain). Degrees 3-4 on the mean best; StudentT no gain; log-sd degree 3-4 no gain.
Final: Normal(quartic mean in x0, exp(quadratic) sd), 8 constants, fit by ML on all 250 rows. In-sample NLL 0.575, CV NLL 0.601.

expected_nll: 0.61
