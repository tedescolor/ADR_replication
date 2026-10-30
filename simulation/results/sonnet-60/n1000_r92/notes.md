# Notes
Data: 1 covariate, y bimodal for x>0.7 (lower branch ~ sinusoidal in x, upper branch rising ~1.3 to 2.5).
Final model: 2-component Gaussian mixture; lower mean = a+b*sin(w x+p), upper mean quadratic in x,
log-sd linear in x for each, mixing weight sigmoid(linear in x). 14 fitted constants (~20 literals total).
Fit by ML (BFGS, 12 random restarts). Validation: 5-fold CV with refit each fold: mean NLL 0.238
(train NLL 0.227). I did not compare other model families (e.g. Student-t components, other mean shapes), so this is the only candidate validated.
expected_nll: 0.24
