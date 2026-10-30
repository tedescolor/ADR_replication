# Notes
Data: y vs x0 in [0,1]. Scatter shows two branches for x0 > ~0.4 (one rising to ~2.5, one dipping/falling), single noisy branch below.
Model: 2-component Gaussian mixture. Component means are polynomials in u=x0-0.5 (deg 4 and 3), log-sd polynomial (quadratic / linear),
mixing weight logistic-linear in u. 16 fitted constants (plus 0.5 offset, 1). Fit by MLE (L-BFGS-B, 2 restarts).
Validation: 5-fold CV on random folds, mean NLL. Degrees (A,B): (3,2) 0.259, (4,2) 0.240, (5,2) 0.235, (6,2) 0.232,
(4,3) 0.206, (5,3) 0.205, (6,3) 0.206, (5,4) 0.207, (4,4) 0.207. Chose (4,3) (simplest at the plateau).
Training NLL of the final fit: 0.187 (optimistic). Folds were also used for choosing degree, so CV is slightly optimistic.
expected_nll: 0.21
