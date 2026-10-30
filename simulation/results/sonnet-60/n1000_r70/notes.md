# Notes
Data: 1 covariate x0 in [0,1]; y has a smooth branch (dips, rises, falls) and a second branch that splits off near x0≈0.5 and rises to ~2.5 (bimodal at high x0), spread varying with x0.
Model: 2-component Normal mixture. Each component mean is a degree-5 polynomial in z=2*x0-1; log-sd is linear in z; mixture weight is sigmoid of a quadratic in z. Fitted by maximum likelihood (L-BFGS-B, several restarts). 24 constants.
Validation: 5-fold CV (fixed permutation), 2 restarts per fold. Per-fold held-out NLL for the final structure: 0.169, 0.195, 0.191, 0.078 (the fifth fold did not finish within the time limit); train NLL 0.146. Earlier noisier CV runs of other structures (cubic means, lower-degree variants) gave worse or unstable results (some due to local optima).
Not compared: non-polynomial/flexible components, other families (e.g. StudentT), because time ran out.
Expected NLL is the mean of the 4 completed folds (~0.158) plus a margin for the missing fold and selection.
expected_nll: 0.17
