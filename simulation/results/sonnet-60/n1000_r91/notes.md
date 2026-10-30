# Notes
Data: y vs x0 in [0,1]. Scatter plot shows a single noisy curve for x<0.5 that splits into two branches
(one rising ~linearly to 2.5, one falling to -1) for x>0.55; noise sd varies with x.
Model: 2-component Gaussian mixture. Means = polynomials in x (quartic and cubic), log-sd = quadratic in x,
mixing weight = sigmoid(linear in x). 18 fitted constants, fit by ML (BFGS, 4 restarts).
Validation: 5-fold CV (shuffled) over degree configs (about 25 tried). Chosen (4,3,2,2): CV NLL 0.178.
Other configs ranged 0.18-0.30; single-scale or linear-sd variants were clearly worse. In-sample NLL 0.160.
Selecting among configs adds slight optimism, so I add a small margin.
expected_nll: 0.19
