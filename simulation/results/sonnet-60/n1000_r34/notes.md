# Notes
Data: 1 covariate x in (0,1), y. Scatter shows two branches: a wiggly lower curve over all x, and an upper branch that appears for x>~0.45 and rises to ~2.5 (bimodal at high x, noise sd small there, larger at low x).
Model: 2-component Normal mixture. Component means are polynomials in (x-0.5) (lower: degree 7, upper: degree 3), log-sd polynomials (lower: quadratic, upper: linear), mixing weight = sigmoid(quadratic in x). 20 constants plus 0.5 shift, fitted by ML (BFGS, multi-start).
Validation: 5-fold CV (random folds, refit each fold) over complexity grid; CV NLL: simple (deg5/2 branch) 0.249, chosen config 0.222; larger configs plateaued/worsened (0.226-0.232). Final fit on all data: train NLL 0.193.
Expected out-of-sample NLL taken from CV (0.222), slightly rounded up.

expected_nll: 0.225
