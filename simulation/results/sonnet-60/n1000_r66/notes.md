# Notes
Data: 1-D x in (0,1). Lower branch is a smooth curve (dip near x=.25, rise to ~0 at x=.7, fall after .8) with x-dependent noise; from x~.65 an upper branch (rising to ~2.4, tight) appears with weight growing in x -> bimodal.
Model: 2-component Normal mixture; lower mean quartic in x, upper mean quadratic, log-sd quadratic/linear, weight sigmoid(linear). 15 fitted constants plus one literal (1 in 1-w). Fit by MLE (BFGS, multi-start).
Validation: 5-fold CV (shuffled) refitting each fold: mean NLL 0.212 (folds 0.12-0.32); in-sample 0.187. Not extensively tuned, so CV estimate is near-honest.
expected_nll: 0.22
