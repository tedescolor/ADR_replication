# Notes
Data: x0 in (0,1); y bimodal for x0>0.7 (upper branch rising to ~2.5, lower branch falling to ~-1); main branch has a cubic-like mean, sd growing with x.
Model: 2-component Normal mixture. Component A: cubic mean, log-linear sd. Component B: quadratic mean, constant sd. Weight of B = sigmoid(linear in x). 12 fitted constants (ML, multi-start BFGS).
Validation: 10-fold CV (shuffled) refitting everything: NLL 0.307; in-sample 0.251. (Single Gaussian baseline ~1.4.) Did not tune further, so little selection bias.
expected_nll: 0.31
