# Notes
Data: 1 covariate x0 in (0,1), y bimodal at high x (two branches: lower one dips then falls, upper one rises).
Model: 2-component Normal mixture; branch 1 mean = quartic in t=x-0.5, log-sd quadratic; branch 2 mean quadratic, log-sd linear;
upper-branch weight = sigmoid(quadratic). 16 fitted parameters + the 0.5 shift (well under 80 constants).
Validation: 5-fold CV (fixed random folds), multi-start L-BFGS ML. Mean held-out NLL:
quartic/quadratic-ish simpler variants 0.25-0.28; chosen (5,3,3,2,3) 0.224; larger (6,3,3,2,3) 0.226, (6,3,3,3,3) 0.228 (no gain).
Final fit on all data: train NLL 0.202; CV NLL 0.224 (used as estimate).
expected_nll: 0.23
