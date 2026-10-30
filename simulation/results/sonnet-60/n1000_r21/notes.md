# Notes
Data: x0 in [0,1]; y has two branches that split near x~0.55 (one rising to ~2.5, one falling to ~-1), so conditional density is bimodal at high x.
Model: 2-component normal mixture; component means are polynomials in x (quartic and cubic), log-sd linear in x, mixing weight = sigmoid(linear in x). 21 constants.
Validation: 5-fold CV (random folds) of ML fits. Compared polynomial degrees (3,2)->0.190, (4,2)->0.153, (4,3)->0.137, (4,3, quadratic log-sd)->0.141; degree 5 fits were unstable (CV 0.53). Chose (4,3,linear sd). Final refit on all data (train NLL 0.115).
Expected NLL is the CV estimate, slightly hedged for the selection among configs.
expected_nll: 0.14
