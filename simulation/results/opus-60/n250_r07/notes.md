# Notes

Data: 250 rows, x0 in (0,1), y. For x < ~0.55 y follows a single smooth curve
(down to ~-0.8 near x=0.2, back to ~0 at 0.55) with noise sd ~0.3 shrinking with x.
For x > ~0.6 y splits into two branches: an upper branch rising from ~0.6 to ~2.5
(about 2/3 of points at high x) and a lower branch drifting from ~0.2 down to ~-0.9;
within-branch noise is small (~0.15).

Model: 2-component Gaussian mixture.
- component 1 (main/lower branch): mean quartic in x, log-sd linear in x;
- component 2 (upper branch): mean quadratic in x, constant sd;
- weight of component 2: sigmoid(linear in x).
Fitted by maximum likelihood (BFGS, several random restarts).

Validation: 10-fold CV repeated twice (different shuffles), refitting everything
per fold; mean held-out NLL per row. Variants compared (dl, du, ds, dw, dsu =
polynomial degrees of lower mean, upper mean, lower log-sd, weight logit, upper log-sd):

| config | train NLL | CV NLL |
|---|---|---|
| (4,2,1,1,0) | 0.131 | 0.185 |
| (3,2,1,1,0) | 0.142 | 0.197 |
| (5,2,1,1,0) | 0.125 | 0.180 |
| (4,2,2,1,0) | 0.129 | 0.187 |
| (4,2,0,1,0) | 0.195 | 0.255 |
| (4,2,1,2,0) | 0.127 | 0.194 |
| (4,2,1,1,1) | 0.122 | 0.177 |
| (4,1,1,1,0) | 0.133 | **0.174** |
| (4,1,1,1,1) | 0.125 | 0.174 |

Choice: (4,1,1,1,0) -- best CV tied with (4,1,1,1,1) but one constant fewer.
Final program: quartic main-branch mean, linear upper-branch mean, log-linear sd for
the main branch, constant sd for the upper branch, logistic weight; 12 constants,
refit on all data (train NLL 0.133).

Expected NLL: CV gave 0.174; selection among ~9 variants biases this slightly
downward, and the bifurcation region with few points adds variance, so I add a small margin.

expected_nll: 0.19
