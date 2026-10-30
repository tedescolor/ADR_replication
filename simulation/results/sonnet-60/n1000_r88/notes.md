# Notes
Data: 1 covariate, y. Mean is nonlinear in x0, noise sd shrinks with x0 (0.4 -> 0.1).
Single-Normal models with polynomial mean/log-sd: 5-fold CV NLL ~0.627. Student-t worse (0.645+).
Residuals had negative kurtosis; a symmetric 2-component mixture gave 0.565; separate-mean
two-branch mixtures (each cubic mean) gave a big drop: ~0.18.
Validation: 5-fold CV (shuffled, seed 0), all parameters refit per fold (several random starts).
Candidates (CV NLL): 2 branches cubic mean, log-sd linear per branch, logistic weight linear: 0.1765;
shared log-sd linear + sd ratio + constant weight (12 params, chosen): 0.1803; quartic means: 0.1786.
Chose the simple one: nearly equal CV, fewer parameters. Final: Mixture of two Normals with cubic
means, shared exp-linear sd (second scaled by 1.074), weights 0.528/0.472. ~25 constants.
In-sample NLL 0.172; CV 0.180.
expected_nll: 0.185
