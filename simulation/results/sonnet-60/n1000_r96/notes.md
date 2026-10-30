# Notes
Data: x0 in [0,1]; y is a nonlinear function of x with a second, upper branch that appears for x>0.6 (bimodal at high x, upper mode ~1.7 -> 2.4).
Model: 2-component Gaussian mixture. Lower mean = quintic in x, upper mean = cubic; log-sd quadratic (lower) / linear (upper); upper weight = sigmoid(quadratic in x). 18 fitted constants + 1.
Fit by maximum likelihood (BFGS, multiple starts) on all data.
Validation: 5-fold CV (random folds), compared polynomial degrees (mean NLL): (5,3,2,2,1) 0.211, (6,3,3,2,1) 0.191, (7,4,2,3,2) 0.187, (6,4,3,3,2) 0.1827 (chosen, ~lowest), (7,4,4,3,2) 0.187, (7,4,3,3,1) 0.198.
Train NLL of final fit 0.166; CV NLL 0.183 (the mild selection over ~13 configs adds a little optimism).
expected_nll: 0.19
