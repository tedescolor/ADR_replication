# Notes
Data: y vs x0 in [0,1]; unimodal for x0<0.5, bimodal for x0>0.6 (upper branch rising to ~2.5, lower falling to ~-1).
Model: 2-component Gaussian mixture; component means cubic in x, log-sd linear in x, weight = sigmoid(quadratic). 15 constants.
Validation: 5-fold CV (shuffled), refit per fold, mean held-out NLL:
cubic means/quad weight/linear logsd 0.2155; quartic means 0.2150; quintic/cubic-weight 0.2177 (worse). Chose the simple cubic one.
Final fit on all data: train NLL 0.193. CV estimate is 0.2155 (slightly pessimistic as final uses all data).
expected_nll: 0.217
