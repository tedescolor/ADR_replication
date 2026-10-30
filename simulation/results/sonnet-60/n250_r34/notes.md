# Notes
Data: y vs x0 shows a smooth nonlinear mean (quintic-like) and, for x0 > ~0.65, two branches
(a lower continuation and an upper rising branch), so a single Normal/StudentT with
polynomial mean and log-sd gets CV NLL 0.714 (best of poly mean deg 3-7, log-sd deg 1-3).
Final: 2-component mixture. Comp A: Normal, quintic mean, log-linear sd. Comp B: Normal, quadratic
mean, constant sd. Weight of B = sigmoid(linear in x0). 17 free constants.
Validation: 5-fold CV (fixed random split), refit with 6 restarts per fold.
CV NLL: A deg3/B lin 0.333, A4/B1 0.315, A5/B1 0.296, A6/B1 0.298, A5/B2 0.285 (chosen).
In-sample NLL 0.215. Expect somewhat above CV estimate due to selection among configs.
expected_nll: 0.30
