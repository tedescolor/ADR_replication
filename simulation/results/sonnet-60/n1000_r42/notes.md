# Notes
Data: 1 covariate x0 in [0,1]. y is unimodal at low x (nonlinear mean, sd ~0.3) and bimodal for x>0.65: a lower branch and an upper branch that rises to ~2.4.
Model: 2-component Gaussian mixture. Component means are polynomials in x (degree 4 and 2), log-sd is linear in x, and the weight of the upper branch is sigmoid(linear in x). 17 constants in total.
Validation: 5-fold CV (shuffled), refitting on each training fold by ML with BFGS, 4 starts.
CV mean NLL: (deg 3,1) 0.260; (4,1) 0.250; (4,2) 0.237; (5,2) 0.235; others unstable. I chose (4,2) for stability. In-sample NLL was 0.222.
expected_nll: 0.24
