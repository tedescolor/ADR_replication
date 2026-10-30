# Notes
Data: x0 in [0,1]; y has a single noisy branch for x<0.5 that splits into an upper (rising to ~2.5) and a lower (falling to ~-1) branch for x>0.55; noise level varies with x.

Model: 2-component Gaussian mixture. Each component has a polynomial mean (degree 4) and a log-sd that is quadratic in x. The mixing logit is quadratic in x. 19 fitted constants, fit by maximum likelihood (L-BFGS, several inits).

Validation: 5-fold CV on the mean held-out NLL, comparing (mean deg, log-sd deg, weight deg):
(3,1,1) 0.208; (4,1,1) 0.210; (4,2,2) 0.203 (chosen); (5,2,2) 0.207; (3,2,2) 0.213; (3,2,1) 0.222; (3,1,2) 0.208; (3,2,3) 0.212.
The differences between the top configurations are small, within fold noise. The in-sample NLL of the final fit is 0.181, which is optimistic.

expected_nll: 0.205
