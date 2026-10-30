# Notes
Data: 1 covariate x0 in [0,1]; y shows two branches for x>0.5 (upper rising to ~2.5, lower falling), single noisy band for x<0.5, noise shrinking with x.
Model: 2-component Gaussian mixture; each component has cubic-polynomial mean and log-sd quadratic in x; mixing weight = sigmoid(quadratic in x). 25 constants (excluding exponent integers counted too, well under 80).
Fit by ML (L-BFGS, multiple starts). Validation: 5-fold CV (same folds for every config), mean held-out NLL:
(mean deg, logsd deg, weight deg): (3,1,1) 0.244; (3,2,2) 0.240; (4,2,2) 0.245. Others (3,1,2),(4,1,2),(5,2,2),(4,2,3) not completed (time).
Chose (3,2,2), refit on all data (train NLL 0.218). Differences between configs are small/within noise; CV estimate used as expected value, nudged up slightly for selection.
expected_nll: 0.245
