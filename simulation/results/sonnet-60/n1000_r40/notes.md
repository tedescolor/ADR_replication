# Notes
Data: y vs x0 in [0,1]. Single branch for x0<0.6 (mean dips then rises); for x0>0.65 y splits into two
well-separated branches (bimodal), the upper one's weight rising with x0.
Model: 2-component Normal mixture. Component means are quartic polynomials in x0, log-sds linear in x0
(separate per component), mixing weight = sigmoid(cubic in x0). 18 constants + a few structural ones.
Fit by ML (BFGS, multiple restarts) on all 1000 rows.
Validation: 5-fold CV (random folds, refit each fold). Mean held-out NLL:
single-sd/quad weight variants 0.263-0.266; linear log-sd + cubic weight, cubic means 0.186;
quartic means 0.181 (chosen); quintic 0.187; quadratic log-sd 0.192.
Train NLL of final fit 0.168; CV 0.181; I add a small margin for selection optimism.
expected_nll: 0.19
