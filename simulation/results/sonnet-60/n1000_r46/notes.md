# Notes

Data: 1 covariate x0 in [0,1], y in about [-1.6, 2.6]. A scatter plot shows one branch for x<0.5 and two branches
(one rising to about 2.5, one falling to about -1) for x>0.6, with the noise sd shrinking as x grows within each branch.

Model: 2-component Gaussian mixture. Component 1 has a quartic mean in x0, component 2 a quadratic mean.
Each component has log-sd linear in x0. The weight of component 2 is sigmoid(linear in x0).
Constants: about 24 (limit 80).

Validation: 5-fold CV (random folds, seed 1), refitting by max likelihood (L-BFGS, several starts) in each fold.
Mean held-out NLL by (mean degree comp1, comp2):
(3,1) 0.255; (4,1) 0.236; (5,1) 0.236; (6,1) 0.236; (3,2) 0.233; (5,2) 0.237; (4,2) 0.227 (best).
A (4,3) run was started, but its result was never recorded; I did not use it.
Chosen (4,2). The full-data fit has training NLL 0.204. Honest out-of-sample estimate is the CV value,
nudged up slightly because of model selection among about 8 candidates.

expected_nll: 0.235
