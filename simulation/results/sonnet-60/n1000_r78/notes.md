# Notes
Data: y vs x0 in [0,1]; nonlinear mean, and for x>0.5 the data split into two branches (rising and falling), so heteroscedastic/bimodal.
Model: 2-component Normal mixture. Component means are polynomials in x (degree 5 and 3), log-sd linear in x, mixture weight = sigmoid(quadratic in x). Fit by ML (BFGS, multi-start) on all data. 17 fitted constants plus a few structural ones, well under 80.
Validation: 5-fold CV (random folds), refitting each fold. CV NLL: linear-weight (4,2,1,1) 0.251; chosen (5,3,1,2) 0.212; (5,3,2,3) 0.221; (6,3,2,3) 0.222; (5,2,2,2) 0.213; (4,3,2,3) 0.214. Train NLL of final 0.175 (optimistic). Chose the simplest config at the CV plateau.
expected_nll: 0.22
