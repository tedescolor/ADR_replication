# Notes
Data: 250 rows, one covariate. Mean of y is non-monotone in x0 (dip near 0.25, steep rise after 0.5); spread grows strongly with x0.
Validation: 10-fold CV (3 repeats for the Normal baselines, 1 for the mixture).
- Normal, cubic mean, log-sd linear: CV NLL 0.759; log-sd quadratic: 0.615 (best among polynomial degrees 3-6; t-dist gave no gain).
- Standardized residuals had excess kurtosis about -1.06 (light tails). A 2-component normal mixture, with a shared cubic mean and quadratic log-sd and component offsets and scales proportional to s(x), gave CV NLL 0.531.
Final: that mixture (constants ~26, within limit). Train NLL 0.457, CV 0.531.
expected_nll: 0.54
