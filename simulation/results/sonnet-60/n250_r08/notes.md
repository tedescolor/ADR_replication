# Notes
Data: y vs x0 shows a nonlinear mean and, for x0>0.7, two branches (a lower one continuing the main curve, an upper one rising to ~2.4).
Model: 2-component Gaussian mixture. Branch 1 mean is a quartic in x, branch 2 mean is a quadratic. Log-linear sd in x for each branch. Upper-branch weight is sigmoid of a quadratic in x. 15 constants in the fitted parameters.
Fitted by ML (BFGS, 6 starts). Validation was 5-fold CV over random splits, refitting each fold.
CV NLL: poly(3,1) 0.310; (4,1) 0.256; (5,1) 0.268; (4,2) 0.226; (5,2) 0.226; (6,1) 0.263. Chose (4,2). Train NLL is 0.166, optimistic.
expected_nll: 0.24
