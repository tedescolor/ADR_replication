# Notes
Data: 1 covariate x0 in [0,1]; y has a nonlinear mean, and for x0 > ~0.45 splits into two branches (rising and falling), with heteroscedastic noise.
Model: 2-component Normal mixture. Component means are polynomials in u=x0-0.5 (deg 4 and deg 2), log-sd polynomials (deg 2 and 1), and the mixing weight is sigmoid(linear in u). 14 fitted constants plus 1 (the 0.5 shift) and the literal 1 in 1-p.
Fit: ML via L-BFGS with multiple random restarts (small ridge penalty).
Validation: 5-fold CV (same fold split for every candidate), restarts in each fold. Mean held-out NLL:
(deg a, deg b, sd-deg a, sd-deg b): (3,1,2,1) 0.312; (3,2,2,1) 0.243; (4,2,2,1) 0.237 (chosen); (4,1,2,1) 0.37 with fewer restarts; (6,2,3,1) 0.395 (overfit / local optima).
In-sample NLL of the final fit: 0.224. The model selection is slightly optimistic, so I added a small margin.
expected_nll: 0.25
