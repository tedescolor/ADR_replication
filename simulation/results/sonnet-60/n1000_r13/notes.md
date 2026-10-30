# Notes
Data: 1 covariate, y is unimodal at low x (mean U-shaped, sd ~0.3-0.5), but for x>0.55 splits into two branches
(lower ~0, upper rising to ~2.5), so conditional density is bimodal.
Model: 2-component Gaussian mixture; component means are polynomials in x (deg 5 lower, deg 2 upper), log-sd linear in x,
weight of upper = sigmoid(cubic in x). Fit by ML (BFGS, multiple starts). ~22 constants.
Validation: 5-fold CV of the fitting procedure. CV NLL: (3,1,2) 0.214; (4,1,2) 0.199; (4,2,2) 0.192; (5,2,3) 0.187;
(6,2,3) 0.186; (5,3,3) 0.192. Chose (5,2,3) (simplest at the plateau). Full-data train NLL 0.163.
expected_nll: 0.19
