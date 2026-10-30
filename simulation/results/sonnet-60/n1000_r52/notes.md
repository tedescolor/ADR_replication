# Notes
Scatter plot: single noisy branch for x<0.45, then two branches (upper rising ~linear, lower curved).
Model: 2-component Normal mixture; each mean is a polynomial in x (lower quartic, upper cubic),
log-sd quadratic (lower) / linear (upper), upper weight = sigmoid(linear). 16 constants. Fit by ML (BFGS, 6 starts).
Validation: 5-fold CV, repeated fits per fold. CV NLL: poly degrees (3,2) 0.218, (4,2) 0.199, (4,3) 0.173, (5,3) 0.177.
Chose (4,3); training NLL 0.151, CV NLL 0.173.
expected_nll: 0.175
