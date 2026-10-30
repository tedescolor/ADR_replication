# Notes

Data: 250 rows, x0 in [0,1]. Scatter shows a smooth wavy lower branch (min ~-0.8 at x~0.2,
~0.4 at x~0.6, ~-0.8 at x=1) with noise sd shrinking from ~0.45 at x=0 to ~0.15 at x=1,
plus a tight linear upper branch (y ~ 0.6 at x=0.6 to ~2.4 at x=1) holding ~30-50% of points for x>0.6.

Model family: 2-component Normal mixture fitted by maximum likelihood (Nelder-Mead + BFGS,
multiple random restarts). Validation: 5-fold CV repeated twice (10 fits/variant),
mean held-out NLL per row.

| variant | params | train NLL | CV NLL |
|---|---|---|---|
| sin+linear lower mean, quad log-sd, step weight (sigmoid*const) | 15 | 0.236 | 0.301 (0.315 cold-start) |
| same, unsaturated logistic weight | 15 | 0.243 | 0.294 |
| cubic lower mean, logistic weight | 14 | 0.250 | 0.298 |
| quartic lower mean, logistic weight | 15 | 0.234 | 0.284 |
| quintic lower mean | 16 | 0.234 | 0.290 |
| quartic, step weight | 15 | 0.223 | 0.285 |
| quartic, log-linear sd1 | 14 | 0.236 | 0.284 |
| quartic, constant sd2 | 14 | 0.234 | 0.280 |
| **quartic, log-linear sd1, constant sd2, logistic weight** | 13 | 0.236 | **0.280** |

Final pick: the last row, since it ties for the best CV with the fewest parameters.
Lower branch: quartic in (x-0.5), sd exp(a+bx). Upper branch: linear, constant sd (~0.157).
Weight: sigmoid(0.027 + 4.57 (x-0.865)). It uses 16 literal constants.
CV folds were warm-started from the full-data fit, so CV may be a little optimistic, and I
selected among ~9 variants. I therefore add a margin of ~0.02.

expected_nll: 0.30
