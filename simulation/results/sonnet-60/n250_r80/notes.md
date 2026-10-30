# Notes
- Data: 1 covariate, y has nonlinear mean and, for x>0.75, two clear branches (upper rising ~1.5->2.5, lower falling ~0 -> -1).
- Single Normal/StudentT with polynomial mean + log-sd polynomial: 10-fold CV NLL ~0.613 (t adds nothing).
- 2-component Normal mixture (comp1: quintic mean, log-sd linear; comp2: quadratic mean, log-sd linear; weight = sigmoid(quadratic in x)); 16 parameters.
  5-fold CV NLL (refit incl. random restarts per fold): 0.194. Other configs 0.195-0.30 (fewer terms worse, more terms no better).
- Final fit on all data (in-sample NLL 0.134); expected NLL discounts for optimism, using CV.
expected_nll: 0.21
