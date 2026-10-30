# Notes
Data: 1 covariate; for x>0.7 y splits into two branches (rising upper, falling lower), heteroscedastic.
Tried (5-fold CV NLL, ML fits by BFGS, multi-start):
- single Gaussian, quartic mean, log-sd linear: 0.778
- 2-component mixture, cubic lower mean: 0.327
- 2-component mixture, quartic lower mean: 0.255 (chosen); quintic: 0.256
Final: Mixture of Normal(quartic mean, exp-linear sd) and Normal(linear mean, exp-linear sd), weight sigmoid(linear in x). 13 fitted constants (+ a few structural literals), fit on all data (train NLL 0.185).
CV estimate 0.255; allowing slight optimism/noise, I report a bit higher.
expected_nll: 0.28
