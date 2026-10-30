# Notes
Data: 1 covariate x0 in [0,1]. Binned look: mean is nonlinear in x, spread grows with x, and for x>0.7 y is clearly bimodal (two branches, one rising to ~2.4, one near -0.5).
Model: 2-component Gaussian mixture. Component means are polynomials in x (degree 4 and 3), log-scales linear in x, mixing weight sigmoid(linear in x). 17 fitted constants plus a few structural ones, well under the limit. Fitted by direct ML (BFGS, multiple restarts).
Validation: 5-fold CV (shuffled), comparing mean polynomial degrees / scale degrees. Mean CV NLL: (3,2,1) 0.234, (4,2,1) 0.250, (4,3,1) 0.219 (chosen), (5,2,1) 0.236, (4,2,2) 0.252. The fold-to-fold spread is about 0.05, so differences between configurations are partly noise. In-sample NLL of the final fit is 0.198.
I expect a bit worse than CV because of selection among configurations and limited restarts.
expected_nll: 0.23
