# Notes
Data: 1 covariate; y has a smooth branch over all x (quartic mean) plus a second, steeply rising branch that appears for x>~0.6 (fork).
Model: 2-component Gaussian mixture; branch A mean = quartic in x, branch B mean = linear in x; weight of B = sigmoid(linear in x); log-sd linear in x per component. 13 fitted constants (+ the 1 in 1-pb).
Fit by maximum likelihood (Nelder-Mead then BFGS, 4 starts). Validation: 5-fold CV, 2 repeats, whole pipeline refit per fold.
CV NLL: (A deg3,B deg1) 0.263; (4,1) 0.245; (4,2) 0.249; (5,2) 0.267; (3,2) 0.266. Chose (4,1). In-sample NLL 0.185 (optimistic).
Expected NLL set slightly above CV for selection bias.
expected_nll: 0.26
