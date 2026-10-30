# Notes
Data: y vs x0 in [0,1]. For x0<~0.45 one branch; above that it splits into two branches (upper rising to ~2.5, lower dipping), noise shrinking with x0.
Model: 2-component Normal mixture. Lower branch mean = quartic in x, upper = quadratic; log-sigma linear in x per component; upper weight = sigmoid(quadratic). 18 constants.
Validation: 5-fold CV, ML fit via BFGS with random restarts. CV NLL: (quartic,linear upper) 0.204; (4,quad) 0.185; (5,quad) 0.194; (6,quad) 0.185. Chose (4,quad), the simplest of the best. Train NLL of final fit 0.171.
Expected NLL is the CV estimate, rounded up slightly.
expected_nll: 0.19
