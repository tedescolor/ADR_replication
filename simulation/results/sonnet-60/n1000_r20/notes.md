# Notes
Data: 1 covariate, y bimodal for x>0.6 (two branches, one rising, one falling); single mode at low x.
Model: 2-component Gaussian mixture; each component has a quartic-polynomial mean in x and log-linear sd;
mixing weight sigmoid(a+b*x) (about 0.5). 16 constants.
Validation: 5-fold CV (shuffled), refit per fold, multi-start MLE. Mean CV NLL by (mean degree, log-sd degree):
(2,1) 0.398, (3,1) 0.175, (3,2) 0.180, (4,1) 0.168, (4,2) 0.177. Chose (4,1); in-sample NLL 0.154.
Optimism of in-sample fit is ~0.014, so estimate is CV-based.
expected_nll: 0.17
