# Notes
Data: y vs x0 shows two branches for x0>~0.45: a rising near-linear upper branch and a sinusoidal lower branch; noise sd shrinks with x0.
Model: 2-component Normal mixture. Lower mean = c + (A+A1 x) sin(w x + phi); upper mean = quadratic in x; both log-sd linear in x; mixing weight = sigmoid(linear in x). 14 fitted constants.
Fit by maximum likelihood (BFGS, several phase starts). Validation: 5-fold CV refitting everything per fold: mean NLL 0.2435 (folds 0.15-0.30); train NLL 0.2252.
Only this one structure was tried (no alternatives compared); the CV gap is small (~0.02) with 14 parameters.
expected_nll: 0.245
