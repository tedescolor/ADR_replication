# Notes

Data: 1000 rows, x0 in [0,1], y in [-1.6, 2.7]. Scatter shows two curves: they overlap / sit
~0.4 apart for x0 < 0.5 and split clearly after x0 ~ 0.6 (upper branch rises to 2.6, lower
bends down to -1.1). Noise shrinks with x0 (~0.3 at x0=0 to ~0.1 at x0=1). About half the
points on each branch everywhere.

Model family: 2-component mixture, means polynomial in t = 2*x0-1, log-sd polynomial in t,
mixing logit polynomial in t; fit by direct MLE (BFGS + Nelder-Mead polish).

Validation: 5-fold CV, repeated with 2 shuffles (seeds 0,1), mean held-out NLL per row.

| means deg | log-sd deg | weight | noise | CV NLL |
|---|---|---|---|---|
| 3 / 4 / 5 | 1 | const | normal | 0.190 / 0.187 / 0.187 (1 shuffle) |
| 6 | 2 | const | normal, separate sd | 0.186 |
| 7 / 8 | 2 | const | normal, separate sd | 0.186 / 0.186 |
| 6 | 1 / 3 | const | normal, separate sd | 0.187 / 0.190 |
| 5 | 2 | linear | normal | 0.185 (1 shuffle, no gain over const) |
| 6 | 2 | const | Student-t df=10 / 5 | 0.191 / 0.202 |
| 6 | 1 / 2 / 3 | const | normal, **shared sd** | 0.184 / **0.181** / 0.185 |
| 7 | 2 | const | normal, shared sd | 0.182 |

Final: degree-6 means for both branches, one shared log-quadratic sd, constant weight
(20 constants). Training NLL 0.158, CV NLL 0.181. Gaussian noise beat Student-t, and tying
the two branches' sd helped. Allowing for some optimism from picking the model by CV:

expected_nll: 0.185
