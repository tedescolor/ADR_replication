# Notes
Data: 1 covariate, nonlinear mean in x0, heteroscedastic; for x0>0.6 y splits into two branches (bimodal).
Model: 2-component Gaussian mixture; each component has cubic mean in x and log-sd linear in x; mixing weight logistic quadratic in x. 15 constants.
Validation: 5-fold CV (random folds, multistart ML fit per fold). CV NLL: cubic/lin-sd 0.197, quartic/lin 0.201, quartic/quad 0.205, quintic/quad 0.202. Chose the simplest/best (cubic, linear log-sd). Train NLL 0.185.
expected_nll: 0.20
