# Notes
Data: 1 covariate x0 in [0,1]; y is unimodal-ish for x<0.5, then splits into an upper branch (rising to ~2.5) and a lower branch (falling to ~-1) for x>0.6, with noise shrinking as x grows.

Model: 2-component Gaussian mixture. Component means are polynomials in u=x-0.5 (degree 5 and 3), log-sd linear in u per component, logistic mixing weight linear in u. Fit by maximum likelihood (L-BFGS, several starts).
Constants: ~30; well under the 80 limit.

Validation: 5-fold CV (shuffled, seed 1), refitting each fold. Degrees (3,2),(4,2),(5,2) -> 0.286, 0.277, 0.276; (5,3) -> 0.232. Training NLL of (5,3) on all data is 0.210. Degree (6,3) gave 0.230 CV, no gain over (5,3) 0.230 (2 restarts), so I kept the smaller one.

Chosen: degrees (5,3). Expected NLL a bit above CV estimate is plausible.

expected_nll: 0.235
