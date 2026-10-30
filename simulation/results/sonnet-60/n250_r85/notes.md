# Notes
Data: 1 covariate, y has a smooth nonlinear mean for x<0.6 and a second upper branch (y~1-2.5) emerging at x>0.65, so the conditional density is bimodal at high x.
Model: 2-component Normal mixture. Component 1: quartic mean in x, log-sd linear in x. Component 2: linear mean, constant sd. Mixing weight = sigmoid(linear in x). 15 constants.
Validation: 5-fold CV (random folds, refit with 6 restarts each). Mean NLL: single-branch-free quadratic-mean 0.439, cubic 0.287, quartic 0.254, quintic 0.256. Chose quartic.
CV NLL 0.254 (train 0.216); expected slightly worse than CV due to selection among few options.
expected_nll: 0.26
