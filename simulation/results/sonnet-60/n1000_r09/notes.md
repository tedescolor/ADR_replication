# Notes
Data: 1 covariate x0 in [0,1]; y has a nonlinear mean, heteroscedastic noise, and for x>~0.5 splits into two branches (upper ~linear rising, lower falling).
Program: 2-component Gaussian mixture. Upper-weight p = 0.493*sigmoid(97*(x-0.252)); component means are polynomials in x (deg 5 lower, deg 3 upper), log-sd quadratic in x for both. 28 constants.
Fit: maximum likelihood (L-BFGS + BFGS) on all 1000 rows.
Validation: 5-fold CV, refitting per fold, over polynomial degree configs (3,2,1,1) .. (6,3,2,2) (mean/mean/logsd/logsd degrees).
CV NLL: 0.2519, 0.2408, 0.2396, 0.2408, **0.2337** (5,3,2,2 chosen), 0.2371, 0.2397 (6,3,2,2). Train NLL of the chosen: 0.2117.
Not tried: other families (StudentT, skew), more than 2 components, or multiple restarts for the final fit beyond 2.
expected_nll: 0.235
