# Notes
Data: 1 covariate, y bimodal for x>0.6 (upper branch rising to ~2.5, lower branch falling), single mode at low x.
Model: 2-component Gaussian mixture; component means are polynomials in x (deg 4 and 3), log-sd linear in x,
mixing weight sigmoid of a quadratic. 17 numeric constants. Fit by ML (BFGS, multiple starts).
Validation: 5-fold CV over several polynomial-degree configs (CV NLL 0.200-0.217; chosen config 0.2007; train NLL 0.186).
Chosen (4,3,1,1,2) as best CV. Note CV is mildly optimistic from config selection.
expected_nll: 0.21
