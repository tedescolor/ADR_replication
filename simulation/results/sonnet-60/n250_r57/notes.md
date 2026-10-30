# Notes
Data: 1 covariate, nonlinear mean in x0 (dip then rise), variance increasing with x0 and larger near 0.
Validation: 3x repeated 5-fold CV over (mean poly degree 3-6) x (log-sd poly degree 1-3) x (Normal vs StudentT), light ridge.
All reasonable choices gave CV NLL ~0.645-0.66; log-sd degree 1 clearly worse (~0.79). StudentT gave no gain.
Final: Normal, quartic mean, quadratic log-sd, fit on all data (8 constants + 1 for x index not counted). CV NLL for this config 0.649; train NLL 0.620.
Slight optimism from selecting among 24 configs, so I add a small margin.

expected_nll: 0.66
