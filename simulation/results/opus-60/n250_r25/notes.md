# Notes

Data: 250 rows, x0 in (0,1). Scatter shows a single noisy branch for x < ~0.48, which then
forks into (a) a tight, nearly linear rising branch (y ~ 5x - 2.36, sd ~0.15) and (b) a noisier
branch continuing a wavy (quartic-like) curve that falls to -0.7 at x=1. Roughly half the
points with x > 0.48 are on the upper branch.

Model family: 2-component Normal mixture
- lower: mean quartic in x, log sd linear in x
- upper: mean linear (quadratic coefficient fitted ~0), constant sd
- weight of upper = c * sigmoid(k (x - x0)), a near-step gate at x0 ~ 0.476, c ~ 0.46

Validation: ML fits (BFGS, multi-start), repeated 10-fold CV (mean held-out NLL per row).
| model | params | train | CV |
|---|---|---|---|
| A: logistic gate a+bx, quad upper | 13 | 0.163 | 0.231 |
| B: step gate w/ ceiling, quad upper | 14 | 0.144 | 0.210 |
| C: B with linear upper | 13 | 0.144 | 0.200 |
| D: B + loglinear upper sd | 15 | 0.142 | 0.219 |
| E: B + quadratic log lower sd | 15 | 0.144 | 0.216 |
| F: B + quintic lower mean | 15 | 0.144 | 0.218 |
| H: sinusoidal lower mean, k fixed 80 | 11 | 0.161 | 0.213 |
| G: C with k fixed 80 | 12 | 0.148 | 0.207 |
| I: G with ceiling fixed 0.5 | 11 | 0.149 | 0.200 |
| K: hard step (threshold by grid), ceiling 0.5 | 10+th | 0.145 | 0.207 |

(CV runs differ in reps 2-3; CV-to-CV noise between variants is ~0.005-0.01, so B..K are
close; C and I are best.)

Extra flexibility (D, E, F) does not pay off out of sample. Final choice: I, the simplest
of the best group (11 fitted constants, 13 literals in model.py): lower branch quartic mean
with log-linear sd, upper branch linear mean with constant sd, upper weight
0.5*sigmoid(80(x-0.4877)). Fitted on all 250 rows; train NLL 0.149, CV NLL 0.200.
The expected value is set a bit above CV to allow for selection among variants.

expected_nll: 0.205
