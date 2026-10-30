# Notes
Data: y vs x0 shows a single noisy curve (dip near x=0.2) that splits at x~0.6 into two branches (upper rising to 2.4, lower falling to -1).
Model: 2-component Normal mixture. Lower branch: quartic mean, log-sd quadratic in x. Upper branch: quadratic mean, log-sd linear. Weight of upper = sigmoid(5.04*(x-0.831)). 15 constants fitted by ML (BFGS, 4 starts).
Validation: 5-fold CV (refit each fold): mean NLL 0.225 (folds 0.18-0.30). In-sample NLL 0.203. Only this program family was explored in depth (no time-consuming search), to limit overfitting with ~15 params.
expected_nll: 0.23
