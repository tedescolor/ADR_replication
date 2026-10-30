# Notes
Data: y vs x0 in [0,1]. Lower branch is a smooth cubic-like curve with heteroscedastic noise. For x0 > ~0.6 a second, tight upper branch (y ~ 1.7-2.5) appears, with weight rising to ~0.5.
Model: 2-component Normal mixture. Lower mean is a cubic in x, lower log-sd is a quadratic, upper mean is a quadratic, upper log-sd is linear, and the weight is sigmoid(quadratic). 15 constants.
Validation: 5-fold CV (refitting by ML each fold). Lower-mean polynomial degree 3/4/5/6 gave CV NLL 0.1790/0.1798/0.1788/0.1792, so I kept degree 3. The full-data train NLL is 0.160.
expected_nll: 0.18
