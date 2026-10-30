# Notes
Data: y vs one covariate x0 in [0,1]. Mean is strongly nonlinear in x0; at x0>0.7 y splits into two branches (one rising to ~2.5, one falling to ~-1), so the conditional density is bimodal there.

Validation: 5-fold CV (random permutation, seed 0), parameters refit on each training fold (3 random starts), NLL on the held-out fold.
- Single Normal, polynomial mean (deg 3-6) and polynomial log-sd (deg 1-3): CV NLL ~0.609. StudentT no better (~0.609).
- Two-component Normal mixture; each component has a polynomial mean and log-linear sd, and the weight is a sigmoid of a polynomial in x0:
  - mean deg 3, sd deg 1, weight deg 1: 0.197
  - mean deg 3, sd deg 1, weight deg 2: 0.193
  - mean deg 4, sd deg 1, weight deg 2: 0.190 (chosen)
  - larger variants (sd deg 2, weight deg 3): 0.193-0.195, no gain.
  The differences among the mixture variants are within noise; I took the best CV one.

Final: the (4,1,2) mixture refit on all 1000 rows. Training NLL 0.174. 17 constants plus 1 for `1 - w`, within the 80 limit; passes the contract check.
Estimate is the CV NLL, with a small allowance for selection among configurations.

expected_nll: 0.195
