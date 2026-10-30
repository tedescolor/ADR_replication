# Notes
Data: 1 covariate, y has two branches for x>0.4 (rising line and a hump going down), single noisy cloud for x<0.4, noise growing with x.
Model: 2-component Gaussian mixture; component means are quartic polynomials in (x-0.5), log-sd quadratic, mixture weight = sigmoid(cubic). 20 fitted constants + offsets, fit by ML (L-BFGS/BFGS, multiple starts).
Validation: 5-fold CV (random folds) of the full fitting procedure. Degrees (mean,sd,weight): (3,1,2) 0.2374, (4,1,2) 0.2389, (4,2,3) 0.2363 (chosen). Other variants were run but their results were not recorded. Full-data training NLL 0.208; CV is optimistic-free but the selection among few configs adds slight bias.
expected_nll: 0.24
