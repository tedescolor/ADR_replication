# Notes
Data: 1 covariate x0 in (0,1), y bimodal for x0>0.6 (two branches: one rising to ~2.5, one falling back to ~-0.9).
Model: 2-component Normal mixture. Component means are polynomials in u=x0-0.5 (deg 4 and 3), log-sds
polynomial (deg 2 and 1), mixing weight sigmoid(a+b*u). Fit by maximum likelihood (L-BFGS, 8 starts).
Constants: ~28 literals, well under 80.
Validation: 5-fold CV (one fixed shuffle), comparing polynomial degrees (mean A, mean B, logsd A, logsd B).
Mean CV NLL: (3,1,0,0) 0.295; (4,2,1,1) 0.232; (5,3,1,1) 0.203; (4,3,1,1) 0.206; (4,3,2,1) 0.200 (chosen);
(4,4,1,1) 0.199; (5,4,1,1) 0.201. Differences among the best are within noise; chose the smaller one.
Full-data training NLL 0.176. Selection among ~14 configs adds slight optimism, so I add a margin to the CV value.
expected_nll: 0.21
