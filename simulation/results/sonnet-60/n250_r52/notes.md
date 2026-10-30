# Notes
Data: y vs x0 is a mixture of two branches: a smooth lower curve (quartic in x0) and, for large x0, an upper
linear branch (y ~ -2.3+4.9x). Weight of upper branch rises with x0 (sigmoid).
Model: 2-component Normal Mixture; means quartic / linear; log-sd linear in x0; weight = sigmoid(linear).
13 constants + few literals. Fitted by ML (L-BFGS, 8 restarts).
Validation: 5-fold CV (random split), compared lower-degree/higher-degree variants:
quartic+linear 0.250, cubic+linear 0.259, quintic 0.262, quartic+quadratic 0.251, sextic 0.361.
Chosen: quartic+linear (CV NLL 0.250; in-sample 0.165, so I expect slight optimism).
expected_nll: 0.26
