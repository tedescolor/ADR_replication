# Notes
Data: 1 covariate x0 in [0,1]. y has a nonlinear mean in x and, for x > ~0.6, splits into two branches
(upper ~1.5-2.5, lower ~-0.1 to -0.8) with increasing spread.

Model: 2-component Normal mixture. Each component has a cubic-polynomial mean in x and
log-sd linear in x; mixing weight is sigmoid(linear in x). 14 constants in total.
Fit by maximum likelihood (L-BFGS, multiple random starts).

Validation: 5-fold CV (random folds, refit with 8 starts per fold). Mean NLL:
mixture cubic/linear (3,1) 0.407; (4,1) 0.416; (3,0) 0.456; (2,2) 0.545; (3,2) 0.392; (2,3) 0.350;
(3,3) **0.316**; (4,3) 0.335. Chose (3,3). Training NLL of final fit (40 starts, all data) = 0.262.
CV is somewhat noisy because of local optima, and model selection among ~11 candidates adds slight optimism,
so I add a margin to the CV figure.

expected_nll: 0.34
