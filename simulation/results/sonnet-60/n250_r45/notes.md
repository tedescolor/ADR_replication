# Notes
Data: 250 rows, one covariate x0 in (0,1). Mean of y is nonlinear in x0 (dips to about -0.8 near 0.25, rises to about 0.8 near 0.9). Spread is heteroscedastic (sd about 0.3 at low x0, about 1.6 at x0 near 1).

Validation: 10-fold CV repeated 3 times (different permutations), mean held-out NLL per row. Each fold refits all parameters by maximum likelihood (BFGS).

Candidates, with (mean polynomial degree, log-sd polynomial degree):
- Normal (3,1) 0.817; (3,2) **0.657**; (4,2) 0.662; (5,2) 0.664; (2,2) 0.776; (3,3) 0.661; (4,3) 0.666
- Student-t (3,2) 0.657, no gain; its df is large.
- Skew-normal (3,2) 0.666, worse than Normal; (3,1) 0.777.

Final: Normal, cubic mean, quadratic log-sd. 7 constants. In-sample NLL is 0.626, CV NLL is 0.657. I use the CV figure as the estimate.

expected_nll: 0.66
