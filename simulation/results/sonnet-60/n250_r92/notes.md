# Notes
Data: y vs x0 has a nonlinear mean (quartic-like dip/peak), sd shrinking with x, and for x>~0.75 a second upper branch
(y~1.6 -> 2.4, rising) alongside the continuing main branch (falling to ~-0.9 at x=1).

Model: 2-component mixture. Main: Normal(quartic poly in x, sd=exp(a+b x)). Upper: Normal(linear in x, const sd 0.139).
Weight of upper = sigmoid(4.40*(x-0.867)). 12 fitted parameters + a few literals, well below limits. Fit by ML (BFGS, 4 restarts).

Validation: 5-fold CV (shuffled, seed 1), refitting each fold. Main-polynomial degree 3/4/5/6 gave CV NLL
0.284/0.275/0.282/0.282, so degree 4 was chosen. Train NLL of the final fit is 0.188 (optimistic).
Not tried: sharper weight functions, other families (t, skew), splines.

expected_nll: 0.28
