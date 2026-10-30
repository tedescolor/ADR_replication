# Notes
Data: y vs x0 shows a smooth branch for x<0.7 that splits into two branches at high x (upper rising to ~2.5, lower falling).
Model: 2-component Gaussian mixture; comp1 mean quartic in x, comp2 mean linear; log-sd linear in x for each; mixing weight = sigmoid(linear in x). 13 parameters (~16 constants).
Validation: 5-fold CV (one split, seed 1), refitting by ML with multi-start BFGS per fold. Mean CV NLL: (deg1,deg2) 4,1: 0.219; 3,1: 0.236; 5,1: 0.256; 3,2: 0.272; 4,2: 0.259; 4,0: 0.340; 2,1: 0.503. Chose 4,1.
Full-data train NLL 0.160; CV is optimistic-free but noisy and model selection among 7 candidates adds slight optimism.
expected_nll: 0.23
