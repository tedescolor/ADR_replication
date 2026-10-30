# Notes
Data: x0 in [0,1]; the mean of y is nonlinear, the spread grows with x0, and for x0 > ~0.6 y splits into two branches (upper rising to ~2.5, lower near 0 to -0.9).
Model: 2-component Normal mixture. Each component has a cubic mean in x0 and log-sd linear in x0; the mixing weight is sigmoid(linear in x0). 16 constants.
Fitted by ML (BFGS, 12 random starts). Validation: 5-fold CV, 2 repeats, refitting each fold.
CV NLL (mean per row): cubic/cubic/const-sd 0.31; quad upper 0.41; linear log-sd 0.25 (3,2) / 0.230 (3,3) / 0.233 (4,4) / 0.240 (5,3); quadratic log-sd 0.245-0.25.
Chosen: (3,3, linear log-sd). In-sample NLL 0.150, CV 0.230; the CV figure is slightly optimistic from model selection, so I add a margin.
expected_nll: 0.25
