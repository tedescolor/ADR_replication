# Notes
Data: x0 in [0,1]. y is a single noisy curve (dipping, sd larger near x=0) for x<0.4, then splits into
two branches (upper rising to ~2.5, lower flat then falling to -1) for x>0.45.
Model: 2-component Gaussian mixture; weight = sigmoid(cubic in u=x-0.5); component means = quartic in u,
log-sds = quadratic in u. Fit by maximum likelihood (L-BFGS, multiple starts). 20 fitted coefficients + a few literals.
Validation: 5-fold CV (random folds), refitting each fold. CV NLL: deg (mean,logsd,weight) 3,2,2: 0.2065; 3,2,3: 0.2058;
4,2,3: 0.2053; 4,3,3: 0.2083; 5,3,3: 0.2053; 5,3,4: 0.2042; 6,3,4: 0.2052. Differences are within noise, so I
picked the simpler (4,2,3). Train NLL of final fit is 0.174 (optimistic).
expected_nll: 0.207
