# Notes
Data: 250 rows, 1 covariate. Mean of y is nonlinear in x0 (dip then rise), and the spread is
heteroscedastic (sd lowest around x0~0.3, large near x0=1).
Validation: 10-fold CV (single shuffle, seed 0), mean NLL, refitting by MLE in each fold.
Candidates (CV NLL): cubic mean + linear log-sd Normal 0.774; + quadratic log-sd 0.644;
quartic/quintic mean 0.653/0.658; sin+linear mean 0.652; cubic log-sd 0.649;
Student-t (df=8) was worse than Normal in every variant; quadratic mean 0.756.
Final: Normal(cubic(x), exp(quadratic(x))), 7 constants, fit by MLE on all data
(train NLL 0.615, CV NLL 0.644).
CV picks among a few candidates, so a slight optimism is possible; I add a small margin.

expected_nll: 0.66
