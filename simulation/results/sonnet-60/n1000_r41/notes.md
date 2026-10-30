# Notes
Data: 1 covariate x0 in [0,1]. The mean of y is nonlinear in x0. For x0 > ~0.6 the conditional density splits into two branches (a rising upper one and a falling lower one), so it is bimodal and the spread grows.

Candidates: a 2-component Normal mixture with polynomial means, log-sd polynomials and a logistic weight (quadratic in x). Fitted by ML (L-BFGS, 3 restarts).
Validation: 5-fold CV (shuffled, seed 1), mean held-out NLL.
(dm = number of mean coefficients per component, ds = number of log-sd coefficients per component)
- dm=4, ds=2: 0.1866
- dm=5, ds=2: 0.1863
- dm=5, ds=3: 0.1764  <- chosen
- dm=6, ds=3: 0.1762 (no gain)
- dm=6, ds=2: 0.1896

Final: refit on all 1000 rows (train NLL 0.159). Program: 19 constants, passes the contract check.
CV is slightly optimistic only through model selection over 5 configs; the gap to the train NLL is ~0.017.

expected_nll: 0.18
