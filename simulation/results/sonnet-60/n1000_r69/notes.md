# Notes
Data: 1 covariate on [0,1]. Scatter shows one branch for x<0.4 and two branches for x>0.5 (one rising to ~2.5, one falling to ~-1), so a conditional mixture of two normals.
Model: 2-component Normal mixture; component means are polynomials in x (cubic and quadratic), log-sd quadratic / linear, mixing weight = sigmoid(linear in x). 15 constants.
Validation: 5-fold CV (two different shuffles), fitting by maximum likelihood (L-BFGS) in each fold.
CV NLL: poly degrees (3,1) 0.28, (4,1) 0.277, (3,2) 0.19, (4,2) 0.19-0.196, (6,2) 0.207. Chose (3,2) as simplest of the best. Training NLL on all data 0.174.
Expected CV NLL is about 0.19 (+- 0.012).
expected_nll: 0.19
