# Notes
Data: x0 in [0,1]. One noisy curve for x<0.45; after that it splits into an upper branch (rising to ~2.4) and a lower branch (falling after 0.7).
Model: 2-component Gaussian mixture. Lower component: quartic mean, log-sd linear in x. Upper component: cubic mean, log-sd linear. Weight of the upper component: sigmoid of a quadratic in x. Fitted by ML (BFGS, 3 starts).
Validation: 5-fold CV over (lower degree 3-6) x (upper degree 1-3). Mean NLL was 0.232-0.254. (4,3) gave 0.235 and (6,3) gave 0.232. I chose (4,3) for fewer constants. Training NLL on all data is 0.219.
Constants: about 30 (limit 80).
expected_nll: 0.24
