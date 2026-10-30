# Notes
Data: 1 covariate x0 in [0,1]. The scatter plot shows two crossing branches: a rising near-linear one
(y ~ 2.5 at x=1) and a humped one (back to -1 at x=1); they overlap for x<0.5.
Model: 2-component mixture of Normals. Branch means are polynomials in t=x-0.5 (degree 4 for the humped
branch, 2 for the rising one). The mixing logit is quadratic in t. Each branch has log-sd linear in t.
15 fitted constants plus 0.5 (about 30 in all). It is fitted by ML (BFGS, multiple starts) in work/fit.py.
Validation: 5-fold CV, refitting per fold, across polynomial degrees (mean, weight logit, log-sd).
Single Gaussian-like or low-degree variants gave CV NLL 0.26-0.34; the chosen (4,2,2,1) gave 0.214; neighbours
(4,2,3,1) 0.214, (4,3,2,1) 0.218, (4,2,2,2) 0.216, so it is on a plateau. Training NLL of the full fit is 0.202.
Some selection optimism is possible, since the config was picked among about 14 on the same folds.
expected_nll: 0.22
