# Notes
Data: y vs x0 shows two branches for x0>~0.65 (lower branch non-monotone, upper branch rising ~1.4 -> 2.4).
Model: 2-component Gaussian mixture. Lower mean = quartic in x, upper mean = linear, log-sd linear in x for each,
mixing weight = sigmoid(linear in x). 16 constants.
Validation: 5-fold CV (one shuffle) of ML fits (6 random restarts each), grid over polynomial degrees (lower 2-5, upper 0-2).
Quartic/linear gave CV NLL 0.335 (cubic/linear 0.339, deg 5/2 0.332 - not worth the extra parameters). Constant-upper-mean or low-degree lower: 0.48-0.57.
In-sample NLL of final fit 0.273; CV is the honest figure, plus a little for degree selection optimism.
expected_nll: 0.35
