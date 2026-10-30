# Notes
Data: 250 rows, one covariate x0 in [0,1]. y has a nonlinear mean in x0 and variance that grows strongly with x0.
Candidates: Normal or StudentT, with polynomial mean (degree 3-6) and log-sd polynomial (degree 1-3), fit by maximum likelihood.
Validation: 10-fold CV (single random split, seed 0), mean NLL per row.
- sd degree 1: ~0.72. sd degree 2: ~0.624. sd degree 3: no gain.
- mean degree 3, 4 and 5 are within 0.002 of each other. StudentT gives no gain over Normal (tails are not heavy).
Final: Normal, cubic mean in (x0-0.5), log-sd quadratic. 7 constants (plus 0.5). Fit on all the data.
CV NLL 0.624; in-sample NLL 0.601. The CV figure is the honest estimate. It was also used to pick the degrees, so it is slightly optimistic.
expected_nll: 0.63
