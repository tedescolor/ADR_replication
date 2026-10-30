# Notes
Data: y vs x0 shows a smooth lower branch (nonlinear mean, sd shrinking with x) and, from x0~0.5, a second linear upper branch (bimodal at high x).
Model: 2-component Normal mixture. Lower: degree-6 polynomial mean, log-sd linear in x. Upper: linear mean, log-sd linear. Upper weight = sigmoid(linear in x). 15 constants.
Validation: 5-fold CV (random folds), refit each fold by ML (BFGS). Mean CV NLL: poly deg 3: 0.273, 4: 0.258, 5: 0.259, 6: 0.252 (upper quadratic 0.257 at deg 4). Chose deg 6, linear upper. In-sample NLL 0.234.
CV optimism ~0.02; expected NLL taken as CV value plus small margin.

expected_nll: 0.255
