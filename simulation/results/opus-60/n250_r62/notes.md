# Notes

Data: 250 rows, x0 in [0,1]. For x < ~0.5, y is a single noisy band around -0.3..-0.8.
Above ~0.55 it splits into two branches: an upper one rising to ~2.5 at x=1 and a lower
one falling to ~-0.9 at x=1, with roughly equal mass.

Model family: 2-component Gaussian mixture; component means are cubic in x, log-sd
polynomial in x, and the mixing weight is logistic-polynomial in x. Fitted by maximum
likelihood (L-BFGS, multiple random starts).

Validation: 5-fold CV, repeated 2-3 times with different shuffles; refit (incl. restarts)
inside every fold.

| spec (deg m1, m2, logsd1, logsd2, logit w) | train NLL | CV NLL |
|---|---|---|
| 3,3,1,1,1 | 0.190 | 0.257 |
| 3,3,0,0,1 | 0.266 | 0.356 |
| 3,3,2,2,2 | 0.179 | 0.303 |
| 2,3,1,1,1 | 0.240 | 0.315 |
| 3,3,1,1,0 (constant weight) | 0.192 | 0.250 (2x5-fold, 4 starts) |

Choice: spec (3,3,1,1,0), refit on all data (13 constants): cubic means, log-linear
sds, constant mixing weight 0.451. It matched/beat the x-dependent-weight version in CV
(0.250 vs 0.257) with one fewer constant. Richer variants overfit (CV worse despite
better train NLL); constant-sd variant underfits.

Expected NLL: the CV estimate (0.250) comes from models fit on 80% of the data, the final
model uses 100%, so it should do slightly better; but CV results vary by fold/shuffle, so
I keep a small margin above the CV value.

expected_nll: 0.26
