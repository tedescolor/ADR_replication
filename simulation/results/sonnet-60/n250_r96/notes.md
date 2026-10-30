# Notes
Data: 250 rows, 1 covariate. Mean of y is nonlinear in x0 (dip near 0.25, rise to ~0.9 by x0=0.8); sd grows strongly with x0 (0.3 to 1.5), non-monotone at low x0.
Candidates, 10-fold CV mean NLL (one random split): quartic mean + log-sd linear: 0.822; quartic mean + log-sd quadratic: 0.671;
sinusoid+linear mean variants: 0.820 / 0.670; Student-t noise gave no gain (0.671).
Final: Normal with quartic-polynomial mean and exp(quadratic) sd, fit by ML on all data (8 constants + literals in the code). Train NLL 0.640.
CV estimate 0.671 (single 10-fold split, so noisy); expected_nll is that CV figure.

expected_nll: 0.675
