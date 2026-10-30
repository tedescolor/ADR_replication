# Notes
Data: 1 covariate x0 in [0,1]. y is unimodal for x<0.4 and splits into two branches (upper rising
roughly linearly to ~2.5, lower curving down to ~-1) for x>0.4, with noise sd shrinking as x grows.

Model: 2-component Normal mixture. Lower mean = quartic in x, upper mean = quadratic, log-sd linear in x
(shared slope, separate offset for the upper branch), upper weight = sigmoid(quadratic). 14 fitted constants
(plus the 1 in `1 - p`), well under the 80 limit. Fit by maximum likelihood (L-BFGS-B, 2 restarts).

Validation: 5-fold CV (seed 1) on the mean NLL, refitting per fold. Compared polynomial degrees
for the means, log-sd and weight logit. CV NLL: single-sd/lower-degree variants 0.20-0.29;
(quartic, quadratic, linear sd, quadratic weight) 0.182; larger variants 0.181-0.187 (no real gain).
Chose the simpler one. Final refit on all data: train NLL 0.169.

Gaussian components are an assumption; no heavier-tail alternatives were tested.

expected_nll: 0.185
