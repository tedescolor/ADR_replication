# Notes
Data: 1 covariate, y shows two branches: a smooth lower curve over all x, plus an upper, roughly linear branch appearing for x>~0.5 (weight rising with x).
Model: 2-component Normal mixture. Lower mean = quintic polynomial in x, log-sd linear in x; upper mean linear, constant sd; mixing weight = sigmoid(linear in x). 13 fitted constants. Fit by penalised ML (L-BFGS, 4 restarts).
Validation: 5-fold CV (seed 1) over lower degree 3-6 and upper degree 1-2. CV NLL: lower deg3 0.291, deg4 0.269, deg5 0.259, deg6 0.259; upper deg2 was worse in each case. Chose lower deg5, upper deg1. Refit on all data (train NLL 0.202).
The CV figure is slightly optimistic since the degree was chosen on the same folds.
expected_nll: 0.27
