# Notes
Data: 1 covariate on [0,1]; y has a single branch for x<0.45 and splits into two diverging
branches (one rising to ~2.5, one falling to ~-1) for x>0.6 -> bimodal conditional density.
Model: 2-component Gaussian mixture; component means are polynomials in x (degree 4 and 3),
log-sds linear in x, mixing weight sigmoid(linear in x). 17 constants.
Validation: 5-fold CV (shuffled) with refits (6 random starts) on candidate degrees:
(mean deg4, 3 | sd lin) 0.208; (5 coef,3) 0.217; (deg4, deg3 -> used) 0.185; larger (deg5/deg3) 0.188;
quadratic log-sd 0.209. Chose the (4,3,linear) config. Final fit on all 1000 rows (train NLL 0.167).
CV on the selected config is slightly optimistic from model selection among 7 candidates.
expected_nll: 0.19
