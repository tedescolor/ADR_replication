# Notes
Data: 1 covariate x0 in [0,1]. For x0 > ~0.65 y splits into two narrow branches
(a lower curve that goes down again, and an upper one rising to ~2.5), so the conditional density is bimodal there.

Model: 2-component Gaussian mixture. Lower-branch mean is a quartic in x, upper-branch mean a quadratic,
the upper-branch weight is sigmoid(quadratic), and each log-sd is linear. 15 fitted coefficients (BFGS, 6 restarts, max likelihood).

Validation: 5-fold CV (shuffled, refit on every fold). Mean held-out NLL:
- single-branch-like / low-capacity variants: 0.31-0.34
- (quartic, quad, quad-weight, linear sds): 0.24
- chosen (quartic, quad, quad-weight, linear log-sd): ~0.197
- larger variants (cubic sds, higher-degree means, cubic weight): 0.195-0.205, no real gain.
The final fit on all data has training NLL 0.177.

expected_nll: 0.20
