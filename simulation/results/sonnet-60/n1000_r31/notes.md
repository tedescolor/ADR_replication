# Notes
Data: 1 covariate x0 in [0,1], y with nonlinear mean (dip near 0.25, rise after) and sd varying with x.
Model: Normal(mean = quartic polynomial in x0, sd = exp(quadratic in x0)); 8 constants, fit by ML on all data.
Validation: 5-fold CV (shuffled) comparing Normal/StudentT/SkewNormal with mean degree 3-5 and log-sd degree 1-3.
Heteroscedasticity (log-sd quadratic) mattered most (0.76 -> 0.61). Student-t and skew-normal gave no gain
(residual kurtosis is about -0.7, skew about 0). Higher polynomial degrees gave <0.003 gain, so kept the simpler model.
CV NLL of chosen spec: 0.609.
expected_nll: 0.61
