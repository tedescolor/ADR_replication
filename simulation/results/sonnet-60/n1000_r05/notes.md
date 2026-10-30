# Notes
Data: 1 covariate on [0,1]; scatter shows two branches for x>0.5 (a linear upper branch, a curved lower branch).
Model: 2-component Gaussian mixture; lower mean = quartic in (x-0.5), upper mean linear, log-sd linear in x for each,
mixing weight sigmoid(linear). About 25 constants (13 fitted parameters plus fixed offsets).
Validation: 5-fold CV (ML fits via L-BFGS, 6 restarts). Mean CV NLL by (lower degree, upper degree):
(3,1) 0.265, (3,2) 0.266, (4,1) 0.250, (4,2) 0.252, (5,1) 0.251. Chose (4,1); higher degrees did not help.
Full-data train NLL 0.233; CV NLL 0.250 is the honest estimate.
expected_nll: 0.255
