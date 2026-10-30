# Notes

Data: 1000 rows, x0 in [0,1]. Scatter shows a single noisy curve for x < ~0.5 (sd ~0.3-0.4,
roughly Gaussian: per-decile skew/kurtosis near 0) that bifurcates after x ~0.5 into an upper
branch (rising almost linearly to ~2.5 at x=1) and a lower branch (falling to ~-0.9), with noise
shrinking as x grows.

Model family: 2-component Gaussian mixture, each component with a polynomial mean in x,
log-sd polynomial in x, and a logit-polynomial mixing weight. Fit by MLE (BFGS, multiple random
restarts).

Validation: 5-fold CV of held-out mean NLL, repeated over 3 different fold splits (seeds) for
the main candidates. Configs (deg mean1, mean2, logsd1, logsd2, logit w):
- (4,4,2,2,2): 0.205-0.213; (5,5,3,3,2): 0.216; (4,4,3,3,2): 0.211
- (3,3,2,2,2): 0.201/0.201/0.209; (2,3,2,2,2): ~0.237 (underfit)
- (3,3,2,1,2): 0.196/0.197/0.203  <- chosen (mean 0.198)
- (3,3,1,1,2): 0.197/0.198/0.203; (3,3,3,1,2): 0.197/0.200/0.207; (3,3,2,0,2): 0.206
Also tried a structured "shared base curve + softplus split" parameterisation: CV ~0.207, no
better and harder to optimise.

Final: (3,3,2,1,2) refit on all 1000 rows (train NLL 0.183), ~30 constants.
Expected held-out NLL ~ CV value (fit on 80% data), slightly optimistic-to-neutral overall.

expected_nll: 0.20
