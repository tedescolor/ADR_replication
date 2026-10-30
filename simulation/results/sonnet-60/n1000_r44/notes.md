# Notes
Data: 1 covariate, y has nonlinear mean in x (cubic-ish), sd growing with x, residuals platykurtic (kurtosis -0.56).
Validation: 2x repeated 5-fold CV, parameters refit within each fold.
- Normal, cubic mean, quadratic log-sd: CV NLL 0.642 (higher-degree means / Student-t no better; linear log-sd 0.753)
- Final: symmetric 2-component normal mixture at mu +/- 0.879*s (common scale 0.587*s, weight sigmoid(-0.287)), mu cubic, log s quadratic in x-0.5. CV NLL 0.598 (train 0.578).
Constants: ~20 (<80).
expected_nll: 0.60
