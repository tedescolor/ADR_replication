# Notes
Data: 1 covariate x0 in [0,1]. Scatter shows a smooth nonlinear curve that splits into two branches for x0>~0.5 (upper rising to ~2.5, lower falling to ~-1); noise sd shrinks with x0.

Model: 2-component Normal mixture. Each component: polynomial mean (lower branch deg 4, upper deg 3), log-sd linear in x0; mixing weight = sigmoid(linear in x0). 15 constants. Fit by direct ML (BFGS, multiple starts).

Validation: 10-fold CV (shuffled), refit on each training fold, mean held-out NLL.
Polynomial degrees (A,B) -> CV NLL: (3,1) .233, (4,1) .223, (5,2) .226, (4,3) .1805, (5,3) .181, (6,3) .181, (5,4) .185, (4,4) .186.
Chose (4,3): simplest of the best. Train NLL .163, CV NLL .1805 (slightly optimistic from selecting among degrees).

expected_nll: 0.19
