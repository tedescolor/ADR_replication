# Notes
Data: 1 covariate x0 in [0,1]; scatter shows two branches for x>~0.5 (a smooth nonlinear lower branch, a linear rising upper branch) and heteroscedastic noise.
Model: 2-component Normal mixture; branch A mean = quartic in x, branch B mean = linear; log-sd of each linear in x; weight of B = sigmoid(linear in x). 13 params fit by ML (BFGS, multi-start).
Validation: 5-fold CV (shuffled). Mean CV NLL: homoscedastic quartic/linear 0.309; +heteroscedastic sd 0.228 (cubic 0.255, quintic 0.227, sextic 0.224, quintic+quadratic B 0.222). Chose quartic for parsimony. In-sample NLL 0.215.
expected_nll: 0.23
