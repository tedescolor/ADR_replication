# Notes
Data: 1 covariate, y has smooth nonlinear mean (dip near x=.2, rise) and, for x>~0.7, splits into two branches (upper ~2.1-2.5, lower declining to -1). A single Gaussian with heteroscedastic sigma gave 10-fold CV NLL ~0.63 (poly mean deg 3-5, t / skew-normal no better).
Final: 2-component Normal mixture. Comp 1: quartic mean, log-sd linear in x. Comp 2: linear mean, constant sd. Weight of comp 2 = sigmoid(linear in x). 12 constants.
Validation: 10-fold CV (2 shuffles), refitting by max likelihood in each fold. CV NLL: 0.210 (final structure); alternatives: quintic 0.212, sd quadratic 0.217, comp2 sd linear 0.208 (not worth it), comp1 const sd 0.267, comp2 quadratic 0.25. Full-data in-sample NLL 0.149 (optimistic).
expected_nll: 0.22
