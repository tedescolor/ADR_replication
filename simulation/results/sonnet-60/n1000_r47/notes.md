# Notes
Data: y vs x0 in [0,1]. Mean is a nonlinear wave in x; for x0 > ~0.7 y splits into two narrow branches (one falling, one rising), with spread growing in x.
Model: 2-component Gaussian mixture. Each component has a quartic-in-x mean and log-sd linear in x. Mixing weight = sigmoid(quadratic in x). 17 constants in all.
Fit by ML (BFGS, 6 random restarts). Validation: 5-fold CV, with all fitting redone inside each fold.
CV mean NLL (mean poly degree/log-sd terms/weight terms): 5/2/3 0.198, 5/2/2 0.199, 4/2/2 0.201, 6/2/3 0.197, 5/3/3 0.198, 6/3/3 0.203. The differences are within noise, so I kept the simplest of the best (5/2/3). Final fit on all data (in-sample NLL 0.184).
expected_nll: 0.20
