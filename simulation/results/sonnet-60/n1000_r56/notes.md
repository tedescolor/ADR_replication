# Notes
Data: y vs x0 in [0,1]; mean dips then rises, spread grows with x0, and for x0>~0.6 y is bimodal (two branches, ~-0.4 and ~2.2 at x0~0.9).
Model: 2-component Gaussian mixture; each component has a cubic mean in (x0-0.5) and log-sd quadratic; mixing logit quadratic. 19 fitted constants plus a few structural ones (about 30 in total), fit by penalised ML (BFGS, multi-start).
Validation: 5-fold CV (shuffled, seed 1), refit per fold. Compared (deg mean, deg logsd, deg logit): 2/1/1 0.49; 3/1/1 0.231; 4/2/2 0.228; 5/2/2 0.235; 3/2/2 0.2205 (chosen); 3/3/2 0.226; 3/2/3 0.228; 4/1/1 0.232. Selecting the best of ~12 variants biases CV slightly optimistic, hence the margin below.
Final fit on all data: train NLL 0.199.
expected_nll: 0.235
