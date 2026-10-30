# Notes
Data: 1 covariate x0 in [0,1]. Mean of y is non-monotone in x0; for x0 > ~0.73 y splits into an upper
branch (rising to ~2.5) and a lower branch (falling to ~-1), so a single-component model is wrong there.

Model: 2-component Normal mixture. Component 1 mean = cubic in x, component 2 mean = quadratic,
log-sd linear in x for each, mixing weight = sigmoid(linear in x). 15 fitted constants (+ 1 for the `1 - q`), well under limits.
Fit by penalised (1e-3 L2) max likelihood, L-BFGS, multiple random starts.

Validation: 5-fold CV (one fixed shuffle), refit incl. restarts on each fold. Mean NLL:
mean degrees (4,2): 0.256; (4,3): 0.224 (chosen); (5,3): 0.229. Final fit on all data: train NLL 0.169.
Limits: only a handful of configurations compared with a single CV split, so the selection is slightly
optimistic; no single-Normal baseline was run (the branching is evident in the data).

expected_nll: 0.24
