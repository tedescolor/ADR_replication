# Notes
Data: x0 in [0,1]. Scatter plot shows a single curve for x<0.45 that splits into two branches
(one rising to ~2.5, one falling to ~-1) for x>0.5, with noise sd shrinking-to-growing pattern.
Model: 2-component Gaussian mixture. Each component has a cubic-polynomial mean and a log-sd that
is linear in x0; the mixing weight is sigmoid(quadratic in x0). 15 fitted constants (+1 for 1-p).
Fit by maximum likelihood (BFGS, several random restarts).
Validation: 5-fold CV (shuffled, seed 1), refitting on each training fold. Mean CV NLL:
cubic/lin-logsd 0.2243 (chosen); cubic/quad 0.2260; quartic/quad 0.2266; quartic/lin 0.2277; quintic/quad 0.2281.
Differences are small, so the simplest was chosen. In-sample NLL of the final fit is 0.203.
The chosen CV score was selected among 5 candidates, so it is slightly optimistic; I add a small margin.
expected_nll: 0.23
