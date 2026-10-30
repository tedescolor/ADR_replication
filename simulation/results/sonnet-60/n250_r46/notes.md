# Notes
Data: 1 covariate, y has nonlinear mean in x0 (dip then rise) and sd growing strongly with x0.
Validation: 10-fold CV x3 repeats, mean held-out NLL.
Tried (mean poly degree, log-sd poly degree): Normal/StudentT/SkewNormal. 
Normal cubic mean + quadratic log-sd: CV 0.540; quartic/quintic no better; Student-t and skew-normal no gain; linear log-sd 0.70.
Final: Normal(cubic(x), exp(quadratic(x))), 7 constants + 0 extra, fit by ML on all data (train NLL 0.514).
CV NLL ~0.54.
expected_nll: 0.55
