# Notes
Data: 1-D x in (0,1). Lower branch has a smooth sinusoidal mean with sd shrinking in x; for x>~0.7 a second, upper branch appears (y rising linearly to ~2.5), so the conditional is bimodal there.
Model: 2-component Gaussian mixture. Lower mean = a + b sin(c x + d), sd = exp(a + b x); upper mean linear in x, constant sd; weight = sigmoid(k (x - x0)). 11 constants fitted by ML (Nelder-Mead + BFGS, multi-start).
Validation: 5-fold CV repeated 2x (refit each fold). Constant-sd variant: 0.302; log-linear sd variant (final): 0.256 (se 0.036). In-sample NLL 0.202.
Not tried further: other families (t, skew), more flexible weight/mean forms.
expected_nll: 0.26
