# Notes

Data: 250 rows, one covariate x0 in [0,1]. Scatter shows a single noisy wiggly curve for
x < ~0.55 which splits into two branches for x > ~0.6: an upper branch rising linearly
to ~2.4 at x=1 and a lower branch that plateaus near 0 then drops to ~-0.9. Noise shrinks with x.

Model family: 2-component mixture with x-dependent gate (logistic of polynomial),
component 1 (main/lower) with polynomial mean, log-polynomial scale (Normal or Student-t),
component 2 (upper) linear mean, log-linear scale. Fitted by maximum likelihood (L-BFGS, multi-start).

Validation: 10-fold CV repeated twice (20 fits per spec), held-out mean NLL.
(Fold fits warm-started from the full-data fit plus 2 random starts — slightly optimistic.)

| spec (deg m1, m2, logs1, logs2, gate, t-tail) | train | CV |
|---|---|---|
| 5,1,2,1,1,Normal | 0.188 | 0.265 |
| 5,2,2,1,1,Normal | 0.184 | 0.273 |
| 6,1,2,1,1,Normal | 0.183 | 0.263 |
| 5,1,2,1,1,t | 0.186 | 0.263 |
| 5,1,1,1,1,t | 0.186 | 0.257 |
| 4,1,2,1,1,t | 0.187 | 0.254 |
| 6,1,2,1,1,t | 0.182 | 0.261 |
| 5,1,2,0,1,t | 0.194 | 0.277 |
| 5,1,2,1,2,t | 0.171 | 0.243 |
| 4,1,2,1,2,t | 0.171 | 0.240 |
| **4,1,1,1,2,t** | 0.171 | **0.236** |

Chosen: quartic main-branch mean with Student-t (df≈16.7) and log-linear scale; linear upper
branch with log-linear scale; quadratic logistic gate (upper-branch weight peaks near x≈0.8 at
~0.5, small elsewhere). 15 fitted constants (+1 literal for the 1-w). Lowest CV and among
the smallest models. Estimate inflated a bit for selection among ~11 specs and warm-start optimism.

expected_nll: 0.25
