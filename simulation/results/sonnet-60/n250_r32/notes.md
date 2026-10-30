# Notes
Data: 1 covariate on [0,1]; mean is nonlinear (dip near x=0.2, rises after 0.5); sd is heteroscedastic (min near 0.4, grows to the right).
Candidates: Normal / StudentT, polynomial mean (deg 2-5) x polynomial log-sd (deg 1-3), fit by ML.
Validation: 2x repeated 10-fold CV. Mean NLL: deg(2,2) 0.725; (3,2) 0.654; (4,2) Normal 0.646 / t 0.638; (5,2) 0.641. Log-sd degree 1 is much worse (0.83+); degree 3 gives no gain.
Differences among (3..5, 2) are within noise, and Student-t gives no reliable gain, so I chose the simpler Normal.
Final: Normal, quartic mean, quadratic log-sd, fit on all data. 8 constants for the fit plus 0 others. Train NLL 0.616.
expected_nll: 0.65
