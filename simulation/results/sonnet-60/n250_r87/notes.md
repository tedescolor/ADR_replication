# Notes
Data: y vs x0 shows a sinusoidal-looking lower branch with sd growing slowly in x, and for x0 > ~0.6 a second, tight, linearly rising branch (y ~ 1.2 -> 2.5), about half of the points there.
Model: 2-component mixture. Lower: Normal(a + b sin(w x + p), exp(s0 + s1 x)); upper: Normal(linear in x, const sd); weight of upper = 0.5*sigmoid(60 (x-0.61)). 13 constants.
Validation: 5-fold CV (refit all parameters per fold, ML via Nelder-Mead+BFGS). Free weight/sharpness version: CV 0.228; fixed-sharpness, weight 0.5 version (final): CV 0.210. In-sample 0.181.
Expected NLL is the CV figure.

expected_nll: 0.21
