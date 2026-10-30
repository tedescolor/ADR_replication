# Notes
Data: y vs x0 is nonlinear in the mean; for x0>~0.6 it splits into two branches (upper rising to ~2.5, lower falling to ~-1), with spread growing in x.
Model: 2-component Normal mixture. Lower-branch mean is a quartic in x-0.5, upper-branch mean is a quadratic. Mixing weight is sigmoid(quadratic in x). Log-scales are linear in x. Fit by ML (BFGS, 6 restarts).
Constants: about 30 literals.
Validation: 5-fold CV (shuffled) over polynomial degree pairs (lower, upper). CV NLL: (3,1) 0.237, (4,1) 0.239, (4,2) 0.223, (5,1) 0.243, (3,2) 0.234, (5,2) 0.224. I chose (4,2). The final fit on all data has in-sample NLL 0.169.
Expected NLL is the CV figure plus a small allowance for selection among 6 candidates.
expected_nll: 0.235
