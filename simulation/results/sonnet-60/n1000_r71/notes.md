# Notes
Data: y vs x0 in [0,1]; one branch for x<0.45, then bifurcates into an upper (rising) and lower (falling) branch with heteroscedastic noise.
Model: 2-component Gaussian mixture; each component has a quartic mean in x and log-sd linear in x; mixture weight = sigmoid(quadratic in x). 21 fitted constants (plus a few structural ones), fit by ML (L-BFGS, 6 inits).
Validation: 5-fold CV (random folds) over candidate (mean deg, log-sd deg, weight deg): (3,1,1) .295, (3,2,1) .247, (4,2,1) .328 (local optimum), (4,2,2) .246, (5,2,2) .245, (4,1,2) .243. Chose (4,1,2): simplest of the best. Full-data train NLL 0.225.
CV is slightly noisy because of local optima; expected NLL taken from CV with a small margin.
expected_nll: 0.25
