# Notes
Data: 250 rows, one covariate x0 in [0,1]. Mean of y is nonlinear in x0 (dips near 0.25, rises after); spread grows strongly with x0.
Validation: 10-fold CV (shuffled, seed 0) of mean NLL; params refit within each fold (small ridge 1e-3).
Candidates (poly degree of mean, poly degree of log-sd; CV NLL):
- Normal 2/1 0.782, 3/1 0.709, 3/2 0.532, 4/2 0.532, 5/2 0.527, 3/3 0.544, 3/4 0.539
- StudentT: no better than Normal at same structure (3/2: 0.534)
- SkewNormal: worse (0.58+)
Final: Normal, cubic mean, log-sd quadratic in x0 (7 fitted constants), fit on all data. Chose the simplest model at the CV plateau.
CV NLL 0.532; slight optimism from model selection, so I add a small margin.
expected_nll: 0.55
