# Notes
Data: 250 rows, one covariate x0 in [0,1]. y mean is non-monotone in x0 (dips near 0.2, rises after 0.5); sd is U-shaped-ish and grows strongly for x0>0.7.
Validation: 10-fold CV (single repeat for most configs; seed-fixed folds), mean held-out NLL.
Candidates (polynomial mean degree km, polynomial log-sd degree ks):
- Normal km=3 ks=1: 0.841; ks=2: 0.700; ks=3: 0.701; ks=4: 0.684
- km=4,5 slightly worse than km=3; km=2 clearly worse (0.76)
- Student-t: no gain over Normal. Skew-normal: worse (0.715-0.72).
Final: Normal(cubic mean, exp(quartic log-sd)), 9 constants, fit by ML on all data. Train NLL 0.655, CV NLL ~0.684. The CV choice among ks was done on the same folds, so slightly optimistic; I add a small margin.
expected_nll: 0.70
