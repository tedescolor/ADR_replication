# Notes
Data: 1 covariate x0 in [0,1], y with nonlinear mean (dip near x~0.3, rise after) and
strongly heteroscedastic noise (sd ~0.3 at x~0.3, ~1.3 at x>0.85).

Validation: 10-fold CV, 3-4 repeats with different shuffles, mean test NLL.
Candidates (mean poly degree / log-sd poly degree / family):
- cubic mean, linear log-sd: 0.81; cubic mean, quadratic log-sd: 0.658 (Normal), 0.658 (StudentT, no gain)
- quartic-sextic mean: 0.66-0.67 (no gain); Fourier mean: 0.664+; quadratic mean: 0.76
- log-sd cubic 0.659, quartic 0.648 (within noise, not adopted)
Final: Normal(cubic(x), exp(quadratic(x))), 7 constants, fit by ML on all 250 rows.
In-sample NLL 0.628; CV NLL ~0.658.

expected_nll: 0.66
