# Notes
Data: y vs x0 in [0,1]. The mean is nonlinear (dip then rise). For x0 > ~0.6 the conditional
distribution turns bimodal (a lower branch and an upper branch near 1.7-2.6), with spread growing in x.

Model: 2-component Gaussian mixture. Branch A has a cubic mean in x and log-linear sd. Branch B has a
linear mean and log-linear sd. The weight of B is sigmoid(linear in x). 12 free constants fit by ML
(L-BFGS, multi-start) on all 250 rows, written into model.py.

Validation: 5-fold CV (shuffled, seed 1), refitting each fold. Mean held-out NLL:
- cubic A / linear B: 0.270 (chosen)
- cubic A / quadratic B: 0.275
- quartic A / linear B: 0.271
The remaining configurations (higher weight-polynomial order, etc.) were not finished because of
time. Training NLL is 0.235, so about 0.035 optimism.

expected_nll: 0.28
