# Notes
Data: 1 covariate x0 in [0,1]. Mean of y is nonlinear (sine-like), and for x0>~0.65 y splits into two branches (lower continues the sine, upper rises to ~2.3).
Model: 2-component Gaussian mixture. Component 1: mean A*sin(w*x+phi)+c, constant sd. Component 2: quadratic mean, log-linear sd. Weight of comp 2 = sigmoid(k*(x-x*)). 12 fitted constants (+1 for the literal 1).
Fit by ML (Nelder-Mead then BFGS, 3 starts).
Validation: 5-fold CV (random, seed 0), refitting each fold. Sine-mean version CV NLL 0.183 (fold range 0.05-0.31); quartic-polynomial mean version 0.237. Train NLL 0.131. Chose sine version.
expected_nll: 0.19
