# Notes
Data: x0 in [0,1]. Single smooth curve for x<~0.66, then splits into an upper (rising) and lower (falling) branch with heteroscedastic noise.
Model: 2-component Gaussian mixture; component means are polynomials (quartic lower, quadratic upper), log-sd linear in x, mixing weight sigmoid(k(x-c)). 14 fitted constants, fit by ML (BFGS, multistart).
Validation: 5-fold CV (random shuffles, two seeds), comparing poly degrees (3..5, 1..3). Best (4,2): CV NLL 0.25 and 0.19 (fold noise large); the full-data training NLL is 0.12, so I expect some optimism.
Chosen (4,2) for lowest CV NLL; unimodal alternatives were much worse.
expected_nll: 0.23
