# Notes
Data: 1 covariate x0 in [0,1], n=250. Mean is nonlinear (dip then rise); spread is small mid-range, large near x=1.
Validation: 3x repeated 10-fold CV, mean held-out NLL. Candidates: Normal / StudentT, polynomial mean (deg 3-6), polynomial log-sigma (deg 1-3), SkewNormal.
Results (CV NLL): deg-1 log-sigma ~0.73-0.74 (bad); deg 2 ~0.543 regardless of mean degree 3-5; deg 3 same within noise; StudentT no gain; SkewNormal 0.554 (worse).
Final: Normal, cubic mean, log-sigma quadratic (7 constants + 0 extra). Chosen as the simplest model at the CV plateau.
Train NLL 0.518; CV NLL 0.543.
expected_nll: 0.55
