# Notes
Data: x0 in (0,1). For x<0.6 y is unimodal, a smooth curve (dip near x=0.25, rising) with sd ~0.3. For x>0.65 y splits into two branches (upper rising to ~2.5, lower falling to ~-1), so the conditional is bimodal.
Model: 2-component Normal mixture. Component A has a quartic mean in x and log-sd linear in x. Component B has a quadratic mean and constant sd. The weight is sigmoid of a quadratic in x. 14 constants in the model.
Validation: 5-fold CV (random folds), mean NLL, with 8 random restarts per fit. Candidates (poly degree of A, B -> CV NLL):
(3,1) 0.230; (3,2) 0.213; (4,1) 0.216; (4,2) 0.209; (5,2) 0.210; (6,2) 0.208; (5,1) and (6,1) 0.278 (bad optimisation).
Chose (4,2): the best CV NLL at low complexity. Final fit is on all data (in-sample NLL 0.191).
The CV estimate is slightly optimistic because I picked the best of 8 configurations.
expected_nll: 0.215
