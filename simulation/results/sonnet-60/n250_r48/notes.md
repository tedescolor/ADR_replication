# Notes
Data: 1 covariate x0 in [0,1]. For x0 > ~0.7 y splits into two branches (upper rising ~1.2->2.5, lower falling ~0.2->-0.8); below that a single curved branch.
Model: 2-component Normal mixture. Lower branch mean = quartic in x, upper = linear, log-sd linear in x for each, mixing weight sigmoid(linear in x). 13 fitted constants (+ few structural ones).
Validation: 5-fold CV (shuffled, seed 1), refitting with 6 random starts per fold. Compared polynomial degrees (lower,upper coef counts):
(3,2) 0.70, (4,2) 0.293, (4,3) 0.331, (5,3) 0.303, (6,3) 0.298, (5,2) 0.274, (6,2) 0.269, (7,2) 0.332, (5,1) 0.529.
Chose (5,2): near-best CV with fewer parameters. Train NLL 0.200 (in-sample, optimistic).
Contract check passes.
expected_nll: 0.29
