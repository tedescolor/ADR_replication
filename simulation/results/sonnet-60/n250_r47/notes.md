# Notes
Data: x0 in [0,1], y nonlinear in x0. For x0 > ~0.7 y splits into two clear branches (upper ~1 to 2.5, lower ~0 to -0.9), so a single heteroscedastic Normal/StudentT (poly mean, poly log-sd) is inadequate: 10-fold CV NLL ~0.684.

Final model: 2-component Normal mixture. Component A mean cubic in u=x0-0.5, B mean quadratic; log-sd linear in u for each; weight of A = sigmoid(linear in u). 13 parameters, all written as constants.
Validation: 10-fold CV, 2 repeats, refitting all parameters (multi-start BFGS ML) in each fold. Compared structures (mean degrees, sd forms): (3,2,1,1) best at CV NLL 0.252; (3,1,1,1) 0.303; (4,2,1,1) 0.262; single Normal/t ~0.684. In-sample NLL 0.199 (optimistic).
expected_nll: 0.26
