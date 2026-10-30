# Notes
Data: y vs x0 shows two branches diverging for x0>0.45 (one rising, one falling), single noisier cloud at low x0.
Model: 2-component Gaussian mixture; each component has cubic mean in x, exp(cubic) sd; weight = sigmoid(cubic in x). 20 constants (Horner form). Fit by ML (L-BFGS, multi-start) on all 1000 rows.
Validation: 5-fold CV (shuffled, seed 1), refit per fold. Mean held-out NLL (mean cubic, sd deg, weight deg):
(2,1,1) 0.507; (3,1,1) 0.229; (3,1,2) 0.222; (3,2,1) 0.237; (3,2,2) 0.2195; (4,2,2) 0.222; (3,3,2) 0.2198; (3,3,3) 0.2176 <- chosen.
Gains between top configs are within noise; the choice is mildly optimistic by selection. Train NLL 0.204.
expected_nll: 0.22
